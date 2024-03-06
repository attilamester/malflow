log() {
    local timestamp=$(date +"%Y-%m-%d %H:%M:%S")
    echo "[$timestamp] [Train dockerless CPU] $1"
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
    -m core.processors.bagnet.train_definitions \
    -tb $1 \
    --checkpoints $1
)
