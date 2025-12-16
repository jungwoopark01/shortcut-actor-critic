CUDA_VISIBLE_DEVICES=0 \
    python main.py \
    --seed=0 \
    --env_name=scene-play-singletask-v0 \
    --offline_steps=500000 \
    --online_steps=500000 \
    --agent=agents/shortcut.py \
    --agent.bc_weight=100 \
    --agent.sc_weight=500 \
    --video_episodes=1 \
    --video_frame_skip=1 \
    --log_interval=5000 \
    --eval_interval=100000