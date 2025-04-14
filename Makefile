.PHONY: all check-python venv deps env run clean

# Default target
all: venv deps env run

# Check if Python is installed
check-python:
	@command -v python3 >/dev/null 2>&1 || { echo "Python3 is not installed. Installing..."; \
	if [ "$(shell uname)" = "Darwin" ]; then \
		brew install python3; \
	elif [ "$(shell uname)" = "Linux" ]; then \
		sudo apt-get update && sudo apt-get install -y python3 python3-pip; \
	else \
		echo "Unsupported OS. Please install Python3 manually."; \
		exit 1; \
	fi; }
	@echo "Python3 is installed: $$(python3 --version)"

# Create virtual environment
venv: check-python
	@echo "Creating virtual environment..."
	@python3 -m venv venv
	@echo "Virtual environment created."

# Install dependencies
deps: venv
	@echo "Installing dependencies..."
	@. venv/bin/activate && pip install -r requirements.txt
	@echo "Dependencies installed."

# Create and setup .env file
env:
	@if [ ! -f .env ]; then \
		echo "Creating .env file from .env_example..."; \
		cp .env_example .env; \
		echo "\nPlease update the following values in your .env file:"; \
		echo "LD_SERVER_KEY - Your LaunchDarkly SDK key"; \
		echo "LD_AI_CONFIG_ID - Your LaunchDarkly AI Config ID"; \
		echo "AWS_REGION - Your AWS region for Bedrock"; \
		echo "AWS_ACCESS_KEY_ID - Your AWS access key ID"; \
		echo "AWS_SECRET_ACCESS_KEY - Your AWS secret access key"; \
		sleep 1; \
		$${EDITOR:-vi} .env; \
	else \
		echo ".env file already exists."; \
	fi

# Run the streamlit app
run:
	@echo "Starting Streamlit app..."
	@. venv/bin/activate && streamlit run ai-chatbot.py

# Clean up
clean:
	@echo "Cleaning up..."
	@rm -rf venv
	@echo "Cleanup complete."