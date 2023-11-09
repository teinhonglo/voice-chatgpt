export PYTHONDONTWRITEBYTECODE=1;
export PYTHONNOUSERSITE=1;

if [ ! -v BACKEND ]; then
    BACKEND=chatgpt
fi

if [ "$BACKEND" == "chatgpt" ]; then
    eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
    conda activate chatgpt
elif [ "$BACKEND" == "clip" ]; then
    eval "$(/share/homes/teinhonglo/anaconda3/bin/conda shell.bash hook)"
    conda activate clip
fi
