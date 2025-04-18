# AI Insurance Chatbot using LaunchDarkly's AI Configs

An AI-powered insurance chatbot application that utilizes LaunchDarkly's AI Config capabilities to dynamically manage AI model configurations and prompts in real-time. This project demonstrates how to build a responsive, personalized customer support experience using LaunchDarkly for AI configuration and AWS Bedrock for LLM capabilities.

## Key Features

- **Dynamic AI Configuration**: Update AI prompts, model parameters, and fallback behaviors in real-time without code changes
- **Personalized Responses**: Customize responses based on user policy information and attributes
- **Performance Monitoring**: Track token usage, latency, and time to first token metrics
- **User Feedback Collection**: Gather and analyze user sentiment on AI responses
- **Customizable User Profiles**: Modify insurance policy details to see personalized responses

## Prerequisites

- Python 3.8+
- AWS account with Bedrock access
- LaunchDarkly account

## Getting Started

### LaunchDarkly Setup

1. **Create a LaunchDarkly Account**
   - Sign up at [LaunchDarkly App](https://app.launchdarkly.com)
   - Create a new project for your insurance chatbot

2. **Create an AI Config**
   - Navigate to AI Configs in your LaunchDarkly project
   - Create a new AI Config
   - Name your configuration (e.g., "Insurance Bot")
   - Set your default AWS Bedrock LLM configuration

3. **Configure your AI Model**
   - Provider: Select "Bedrock" 
   - Model: Choose "anthropic.claude-v2:1" (or preferred Bedrock model)
   - Parameters:
     - Temperature: 0.7 (adjust for creativity vs determinism)n
     - Top P: 0.9
     - Max Tokens: 200
   - System Prompt: Define your base insurance assistant prompt (example provided in the `prompt.txt` file included in this repository)

4. **Copy Keys for Application Integration**
   - Copy your LaunchDarkly SDK Server Key
   - Copy your AI Config ID

### AWS Bedrock Setup

1. **Ensure AWS Bedrock Access**
   - Verify your AWS account has Bedrock enabled
   - Request access to Claude and other required models if needed
   - Create IAM user with Bedrock access permissions

2. **Gather AWS Credentials**
   - Create or use an existing AWS access key and secret
   - Note your preferred AWS region where Bedrock is available (e.g., us-west-2)

### Application Setup

1. **Clone the Repository**
   ```bash
   git clone <repository-url>
   cd InsuranceBot
   ```

2. **Run the Setup Script**
   ```bash
   make
   ```
   This will:
   - Create a virtual environment
   - Install dependencies
   - Set up the environment file
   - Prompt you to fill in your credentials

3. **Update Environment Variables**
   Edit the created `.env` file with:
   ```
   LD_SERVER_KEY='your-launchdarkly-sdk-key'
   LD_AI_CONFIG_ID='your-ai-config-id'
   AWS_REGION='your-bedrock-region'
   AWS_ACCESS_KEY_ID='your-aws-access-key'
   AWS_SECRET_ACCESS_KEY='your-aws-secret-key'
   ```

4. **Run the Application**
   ```bash
   make run
   ```
   The application will start on http://localhost:8501

## Usage

1. **Interact with the Chatbot**
   - Ask questions about insurance policies
   - Inquire about coverage details
   - Request policy information

2. **Modify User Profile**
   - Use the sidebar to update insurance policy details
   - See how responses change based on profile attributes

3. **Provide Feedback**
   - Rate responses with thumbs up/down 
   - View metrics in LaunchDarkly dashboard

## Real-time Configuration Updates

You can update the AI behavior in real-time through LaunchDarkly:

1. Modify system prompts to change assistant personality
2. Adjust temperature to control response creativity
3. Update max tokens to control response length
4. Create targeted variations based on user attributes (policy type, coverage level, etc.)

## Learn More

- [LaunchDarkly AI Config Documentation](https://launchdarkly.com/docs/home/ai-configs)
- [AWS Bedrock Documentation](https://aws.amazon.com/bedrock/)
- [Streamlit Documentation](https://docs.streamlit.io/)

## License

This project is licensed under the terms included in the LICENSE file.
