export PYTHONNOUSERSITE=1
export WANDB_DISABLED=true
export WANDB_MODE=offline
#export PYTHONPATH="."
export PYTHONPATH=$(pwd)

if [ ! -v BACKEND ]; then
  BACKEND="openai"
fi

#eval "$(conda shell.bash hook)"
eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"

if [ "$BACKEND" == "openai" ]; then
    CUDA_DIR=/usr/local/cuda-11.6
    conda activate voice-chatgpt-openai
elif [ "$BACKEND" == "local" ]; then
    CUDA_DIR=/usr/local/cuda-12.4
    conda activate voice-chatgpt-local
else
    CUDA_DIR=/usr/local/cuda-11.6
    conda activate voice-chatgpt-openai
fi

if [ -d $CUDA_DIR ]; then
    export PATH=$CUDA_DIR/bin:$PATH
    export LD_LIBRARY_PATH=$CUDA_DIR/lib64:$LD_LIBRARY_PATH
fi
