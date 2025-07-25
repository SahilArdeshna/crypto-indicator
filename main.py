# Entry point for the BTC alert script
from scheduler.job_runner import run
from dotenv import load_dotenv

load_dotenv()

if __name__ == '__main__':
    run()
