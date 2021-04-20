import os
import sys
import time
import glob
import numpy as np
import pickle
import torch
import logging
import argparse
import functools
import random

from maskrcnn_benchmark.modeling.detector.generalized_rcnn import GeneralizedRCNN
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.logger import setup_logger
from maskrcnn_benchmark.utils.miscellaneous import mkdir, save_config
from maskrcnn_benchmark.utils.comm import synchronize, get_rank, get_world_size, all_gather
from tools.train_net import train, run_test
import torch.distributed as dist
import copy

sys.setrecursionlimit(10000)

print=functools.partial(print,flush=True)
choice=lambda x:x[np.random.randint(len(x))] if isinstance(x,tuple) else choice(tuple(x))

class search_config:
    blocks_keys = np.arange(0, cfg.AUTOAUG.NUM_CHOICES).tolist()
    nr_layer=cfg.AUTOAUG.LIST_LENGTH
    states=[cfg.AUTOAUG.NUM_CHOICES]*nr_layer
    max_epochs = 10
    select_num = 10
    population_num = 50
    mutation_num = 20
    m_prob = 0.1
    crossover_num = 20

def reduce_loss_scale(loss_scale):
    world_size = get_world_size()
    if world_size < 2:
        return loss_scale
    with torch.no_grad():
        dist.reduce(loss_scale, dst=0)
        if dist.get_rank() == 0:
            loss_scale /= world_size
    return loss_scale

class EvolutionTrainer(object):
    def __init__(self,cfg, logger, distributed):

        self.log_dir=cfg.OUTPUT_DIR
        self.checkpoint_name=os.path.join(self.log_dir,'checkpoint.brainpkl')

        self.memory=[]
        self.candidates=torch.Tensor([[-1]*search_config.nr_layer]*search_config.population_num).long().cuda()
        self.vis_dict={}
        self.keep_top_k = {search_config.select_num:[],50:[]}
        self.epoch=0
        self.logger = logger
        self.distributed = distributed
        self.results_scale_baseline = [0.204, 0.394, 0.483]
        self.prob_idx = [0, 2] + [(5 + 3*i) for i in range(2*5)]

    def save_checkpoint(self):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        info={}
        info['memory']=self.memory
        info['candidates']=self.candidates if isinstance(self.candidates, list) else self.candidates.cpu().long().tolist()
        info['vis_dict']=self.vis_dict
        info['keep_top_k']=self.keep_top_k
        info['epoch']=self.epoch

        with open(self.checkpoint_name, 'wb') as fid:
            pickle.dump(info, fid, pickle.HIGHEST_PROTOCOL)
        self.logger.info('save checkpoint to %s'%self.checkpoint_name)

    def load_checkpoint(self):
        if not os.path.exists(self.checkpoint_name):
            return False
        with open(self.checkpoint_name, 'rb') as fid:
            info = pickle.load(fid)

        self.memory=info['memory']
        self.candidates = torch.Tensor(info['candidates']).long().cuda()
        self.vis_dict=info['vis_dict']
        self.keep_top_k = info['keep_top_k']
        self.epoch = info['epoch']

        self.logger.info('load checkpoint from %s'%self.checkpoint_name)
        return True

    def update_top_k(self,candidates,*,k,key,reverse=False):
        assert k in self.keep_top_k
        self.logger.info('select ......')
        t=self.keep_top_k[k]
        t+=candidates
        t.sort(key=key,reverse=reverse)
        self.keep_top_k[k]=t[:k]

    def evaluate_single_aug(self, cand, local_rank):
        file_dir = ''
        for i in cand:
            file_dir += str(i)
        cfg.OUTPUT_DIR = os.path.join(self.log_dir, file_dir)

        mkdir(cfg.OUTPUT_DIR)

        output_config_path = os.path.join(cfg.OUTPUT_DIR, 'config.yml')
        self.logger.info("Saving config into: {}".format(output_config_path))
        # save overloaded model config in the output directory
        save_config(cfg, output_config_path)

        model, loss_scale_hist = train(cfg, local_rank, self.distributed, search=self.logger)

        results = run_test(cfg, model, self.distributed)
        results_scales = []
        if not results is None:
            results_bbox = results[0].results['bbox']
            results_scales = [results_bbox['APs'], results_bbox['APm'], results_bbox['APl']]

        if self.distributed:
            loss_scale_hist = reduce_loss_scale(loss_scale_hist)

        return loss_scale_hist, results_scales

    def stack_random_cand(self,random_func,*,batchsize=10):
        while True:
            cands=[random_func() for _ in range(batchsize)]
            for cand in cands:
                if cand not in self.vis_dict:
                    self.vis_dict[cand]={}
                info=self.vis_dict[cand]

            for cand in cands:
                yield cand

    def random_can(self,num):
        self.logger.info('random select ........')
        candidates = []
        cand_iter=self.stack_random_cand(
            lambda:tuple(np.random.randint(i) for i in search_config.states))
        while len(candidates)<num:
            cand=next(cand_iter)

            candidates.append(cand)
            self.logger.info('random {}/{}'.format(len(candidates),num))

        self.logger.info('random_num = {}'.format(len(candidates)))
        return candidates

    def get_mutation(self,k, mutation_num, m_prob):
        assert k in self.keep_top_k
        self.logger.info('mutation ......')
        res = []
        iter = 0
        max_iters = mutation_num*10

        def random_func():
            cand=list(choice(self.keep_top_k[k]))
            for i in range(len(search_config.states)):
                if np.random.random_sample()<m_prob:
                    cand[i]=np.random.randint(search_config.states[i])
            return tuple(cand)

        cand_iter=self.stack_random_cand(random_func)
        while len(res)<mutation_num and max_iters>0:
            cand=next(cand_iter)
            res.append(cand)
            self.logger.info('mutation {}/{}'.format(len(res),mutation_num))
            max_iters-=1

        self.logger.info('mutation_num = {}'.format(len(res)))
        return res

    def get_crossover(self,k, crossover_num):
        assert k in self.keep_top_k
        self.logger.info('crossover ......')
        res = []
        iter = 0
        max_iters = 10 * crossover_num
        def random_func():
            p1=choice(self.keep_top_k[k])
            p2=choice(self.keep_top_k[k])
            return tuple(choice([i,j]) for i,j in zip(p1,p2))
        cand_iter=self.stack_random_cand(random_func)
        while len(res)<crossover_num and max_iters>0:
            cand=next(cand_iter)
            res.append(cand)
            self.logger.info('crossover {}/{}'.format(len(res),crossover_num))
            max_iters-=1

        self.logger.info('crossover_num = {}'.format(len(res)))
        return res

    def train(self, local_rank):
        if local_rank == 0:
            self.logger.info('population_num = {} select_num = {} mutation_num = {} '
                  'crossover_num = {} random_num = {} max_epochs = {}'.format(
                    search_config.population_num, search_config.select_num, search_config.mutation_num,
                    search_config.crossover_num,
                    search_config.population_num - search_config.mutation_num - search_config.crossover_num,
                    search_config.max_epochs))

            if not self.load_checkpoint():
                self.candidates = self.random_can(search_config.population_num)
                self.save_checkpoint()

        while self.epoch<search_config.max_epochs:
            self.logger.info('epoch = {}'.format(self.epoch))

            if isinstance(self.candidates, list):
                self.candidates = torch.Tensor(self.candidates).long().cuda()

            if self.distributed:
                dist.broadcast(self.candidates, 0)

            self.candidates = [tuple(cand.tolist()) for cand in self.candidates]

            loss_scale_hists = []
            results_scales = []
            for cand in self.candidates:
                synchronize()
                cfg.AUTOAUG.LIST = cand
                loss_scale_hist, results_scale = self.evaluate_single_aug(cand, local_rank)
                loss_scale_hists.append(loss_scale_hist)
                results_scales.append(results_scale)

            self.epoch += 1
            if local_rank>0:
                continue
            self.logger.info('Evaluation finish')
           
            for i, cand in enumerate(self.candidates):
                loss_hist = copy.deepcopy(loss_scale_hists[i])
                loss_hist /= loss_hist.sum()
                err = loss_hist.std()
                for j, result_s in enumerate(self.results_scale_baseline):
                    if results_scales[i][j] < result_s:
                        self.logger.info('Punishment for sarcrificing other scales : %s (baseline: %s) in %d th scale of %s.'%(str(copy.deepcopy(results_scales[i])), str(self.results_scale_baseline), j, str(cand)))
                        err *= (result_s/results_scales[i][j])

                # A regularization to avoid probabilities decay to zero.
                l_prob = (9 - np.array(cand)[self.prob_idx].mean()) * 1e-2
                err += l_prob
                self.vis_dict[cand]['err'] = err
                self.vis_dict[cand]['loss_hist'] = str(loss_scale_hists[i])

            self.memory.append([])
            for cand in self.candidates:
                self.memory[-1].append(cand)
                self.vis_dict[cand]['visited'] = True

            self.update_top_k(self.candidates,k=search_config.select_num,key=lambda x:self.vis_dict[x]['err'])
            self.update_top_k(self.candidates,k=50,key=lambda x:self.vis_dict[x]['err'] )

            self.logger.info('epoch = {} : top {} result'.format(self.epoch-1, len(self.keep_top_k[50])))
            for i,cand in enumerate(self.keep_top_k[50]):
                self.logger.info('No.{} {} Top-1 err = {} loss hist = {}'.format(i+1, cand, self.vis_dict[cand]['err'], self.vis_dict[cand]['loss_hist']))
                ops = [search_config.blocks_keys[i] for i in cand]
                self.logger.info(ops)

            mutation = self.get_mutation(search_config.select_num, search_config.mutation_num, search_config.m_prob)
            crossover = self.get_crossover(search_config.select_num,search_config.crossover_num)
            rand = self.random_can(search_config.population_num - len(mutation) -len(crossover))

            self.candidates = mutation+crossover+rand

            self.save_checkpoint()

        synchronize()
        self.logger.info(self.keep_top_k[search_config.select_num])
        self.logger.info('finish!')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)

    args=parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    distributed = num_gpus > 1

    cfg.merge_from_file(args.config_file)
    cfg.AUTOAUG.SEARCH = True
    cfg.MODEL.WEIGHT = cfg.AUTOAUG.FT_WEIGHT
    cfg.SOLVER.MAX_ITER = cfg.AUTOAUG.FT_ITERS
    cfg.SOLVER.BASE_LR = cfg.AUTOAUG.FT_LR

    if distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://"
         )
        synchronize()

    mkdir(cfg.OUTPUT_DIR)
    logger = setup_logger("aug_search", cfg.OUTPUT_DIR, args.local_rank)
    logger.info(search_config)

    t = time.time()

    trainer=EvolutionTrainer(cfg, logger, distributed)

    trainer.train(args.local_rank)
    logger.info('total searching time = {:.2f} hours'.format((time.time()-t)/3600))


if __name__ == '__main__':
    try:
        main()
        os._exit(0)
    except:
        import traceback
        traceback.print_exc()
        time.sleep(1)
        os._exit(1)

