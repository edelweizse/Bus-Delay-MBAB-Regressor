from processing.processor import DataProcessor
from config import Config

def main():
    config = Config()
    processor = DataProcessor(config)

    processor.process()

if __name__ == "__main__":
    main()