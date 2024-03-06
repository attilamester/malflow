log() {
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [Train dockerless GPU] $1"
}

if [ $# -ne 1 ]; then
    log "Usage: $0 <path-to-tb-log-dir>"
    exit 1
fi

./download_dependencies.sh main.py

(
  export $(cat .env | xargs)
  cd ../../../ && \
  python3 -m core.processors.bagnet.main \
    --multiprocessing-distributed --rank=0 --world-size=1 --dist-url='tcp://localhost:29500'\
    -m core.processors.bagnet.train_definitions \
    -tb $1 \
    --checkpoints $1
)
