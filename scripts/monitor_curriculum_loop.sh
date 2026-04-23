#!/bin/bash
cd /home/muchenxu/humanoid-rollerskating-skater-joystick
LOG=logs/amp_curriculum_v1.log
HIST=logs/monitor_curriculum_v1.json
INTERVAL=600  # 10 minutes

echo "=== Starting curriculum monitoring loop ==="
echo "Log: $LOG"
echo "History: $HIST"
echo "Check interval: ${INTERVAL}s"
echo ""

while true; do
  if [ ! -f "$LOG" ]; then
    echo "[$(date)] Log file not found, waiting..."
    sleep 60
    continue
  fi

  PYTHONPATH=src:. python scripts/monitor_training.py --log "$LOG" --history "$HIST" 2>&1
  
  # Check curriculum stage from latest log output
  STAGE=$(grep "Curriculum/stage:" "$LOG" | tail -1 | awk '{print $NF}')
  echo "  [EXTRA] Current curriculum stage: $STAGE"
  echo ""
  
  # Check if training is still running
  if ! pgrep -f "amp_curriculum_v1" > /dev/null 2>&1; then
    echo "[$(date)] Training process not found - stopped or completed."
    break
  fi
  
  sleep $INTERVAL
done
