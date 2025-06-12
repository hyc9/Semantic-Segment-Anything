cd /chengyu/MLLM/Semantic-Segment-Anything && \
echo "===== Git Pull Start =====" && \
git pull && \
echo "===== Git Pull End =====" && \
source activate ssa && \
python scripts/semantic_mask_pipeline_lc.py --world_size 8 

cd /chengyu/MLLM/Semantic-Segment-Anything && echo "===== Git Pull Start =====" && git pull && echo "===== Git Pull End =====" && source activate ssa && python scripts/semantic_mask_pipeline_lc.py --world_size 9
