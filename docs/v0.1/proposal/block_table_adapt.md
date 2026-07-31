# vllm/vllm-ascend框架适配

重新思考SFA的适配逻辑

原有的lightning indexer->SFA的逻辑下，会使用到topk_indices和block table进行计算以在SFA计算过程中寻址
我们现在需要做紧密的block排布以将卸载后驻留HBM的kvcache量压缩下去，想完成一个完全在框架侧实现的适配方案，从这几方面调研：
1. 给我们新的cache逻辑划分的pin住的block
2. 输入给SFA的kv_cache的形状变化
3. 使用专门的block table适应寻址逻辑