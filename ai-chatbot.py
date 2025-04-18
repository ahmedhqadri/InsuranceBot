"""
LaunchDarkly AI Insurance Chatbot

This file demonstrates the integration of LaunchDarkly's AI configuration capabilities with
AWS Bedrock for an insurance chatbot application.

STRUCTURE:
1. LaunchDarkly AI Configuration
2. AWS Bedrock Integration 
3. Main Application
4. Helper Functions
"""

import os
import json
import logging
import time
import random
import dotenv
import streamlit as st
import boto3
from botocore.exceptions import ClientError
from typing import Dict, List, Optional, Any, Generator, Tuple, Union
from datetime import datetime

# LaunchDarkly imports
import ldclient
from ldclient import Context
from ldclient.config import Config
from ldai.client import LDAIClient, AIConfig, ModelConfig, LDMessage, ProviderConfig
from ldai.tracker import FeedbackKind

# Set up logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables
dotenv.load_dotenv()

# Image Constants for UI
if os.path.exists("./src/images"):
    MAIN_LOGO = "./src/images/launchdarkly_logo_black.png"
else:
    MAIN_LOGO = "./images/launchdarkly_logo_black.png"

#######################################################################
###                                                                 ###
###              SECTION 1: LAUNCHDARKLY AI CONFIGURATION           ###
###                                                                 ###
#######################################################################

class LaunchDarklyClient:
    """Main LaunchDarkly client wrapper that handles LD and LDAI operations."""
    
    def __init__(self, server_key: str, ai_config_id: str = "bedrock-config"):
        """
        Initialize the LaunchDarkly client.
        
        Args:
            server_key: LaunchDarkly SDK key
            ai_config_id: The AI configuration ID to use
        """
        # Initialize LD client
        ldclient.set_config(Config(server_key))
        self.ld_client = ldclient.get()
        self.ai_client = LDAIClient(self.ld_client)
        self.ai_config_id = ai_config_id    
    
    def get_ai_config(self, user_context: Context, variables: Dict[str, Any]) -> Tuple[AIConfig, Any]:
        """
        Get the AI configuration for a specific user context.
        
        Args:
            user_context: LaunchDarkly user context
            variables: Variables to pass to the AI configuration including conversation history
            
        Returns:
            Tuple containing the AI config and a tracker object
        """
        try:
            # Create a fallback configuration for when the LaunchDarkly service is unavailable
            fallback_value = self.get_fallback_config()
            
            # Get the configuration and tracker
            config, tracker = self.ai_client.config(
                self.ai_config_id, 
                user_context, 
                fallback_value, 
                variables
            )
            logger.info("AI Config context updated")
            
            self.print_box("MODEL", vars(config.model))
            self.print_box("MESSAGES", config.messages)
            
            return config, tracker
        except Exception as e:
            logger.error(f"Error getting AI config: {e}")
            return self.get_fallback_config(), None
    
    def get_fallback_config(self) -> AIConfig:
        """Return a fallback configuration for when LaunchDarkly is unavailable."""
        return AIConfig(
            enabled=True,
            model=ModelConfig(
                name="anthropic.claude-v2:1",
                parameters={
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "max_tokens": 2000
                },
            ),
            messages=[
                LDMessage(role="system", content="You are a helpful insurance assistant. Maintain context from the conversation history when responding.")
            ],
            provider=ProviderConfig(name="bedrock"),
        )
    
    def send_feedback(self, tracker, is_positive: bool) -> None:
        """
        Send user feedback to LaunchDarkly.
        
        Args:
            tracker: The LaunchDarkly tracker object
            is_positive: Whether the feedback is positive or negative
        """
        if is_positive:
            tracker.track_feedback({"kind": FeedbackKind.Positive})
        else:
            tracker.track_feedback({"kind": FeedbackKind.Negative})
        logger.info(f"Feedback sent: {'positive' if is_positive else 'negative'}")
    
    def print_box(self, title, content):
        """Print content in a styled box for terminal output."""
        import shutil
        
        # Get terminal width
        terminal_width = shutil.get_terminal_size().columns
        max_content_width = terminal_width - 4  # Account for box borders and padding
        
        # Convert content to list of strings if it's not already a list
        content_lines = content if isinstance(content, list) else [content]
        content_str_lines = [str(item) for item in content_lines]
        
        # Calculate initial width based on content and title
        width = min(max(len(title), max(len(line) for line in content_str_lines)) + 4, terminal_width)
        
        # Wrap long content lines to fit terminal
        wrapped_lines = []
        for line in content_str_lines:
            if len(line) > max_content_width:
                # Simple wrapping - split at max width
                for i in range(0, len(line), max_content_width):
                    wrapped_lines.append(line[i:i+max_content_width])
            else:
                wrapped_lines.append(line)
    
        # Print the box
        print('┌' + '─' * (width - 2) + '┐')
        print(f'│ {title[:max_content_width].ljust(width - 4)} │')
        print('├' + '─' * (width - 2) + '┤')
        
        for line in wrapped_lines:
            print(f'│ {line[:max_content_width].ljust(width - 4)} │')
        
        print('└' + '─' * (width - 2) + '┘')


#######################################################################
###                                                                 ###
###                 SECTION 2: AWS BEDROCK INTEGRATION              ###
###                                                                 ###
#######################################################################

class BedrockClient:
    """Client for AWS Bedrock service with generative AI capabilities."""
    
    def __init__(self, region_name: str = None):
        """
        Initialize the Bedrock client.
        
        Args:
            region_name: AWS region name, defaults to environment variable
        """
        self.region_name = region_name or os.getenv("AWS_REGION")
        # Add timeouts to prevent hanging requests
        self.client = boto3.client(
            service_name='bedrock-runtime', 
            region_name=self.region_name,
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    
    def stream_conversation(self,
                    bedrock_client,
                    model_id,
                    messages,
                    system_prompts,
                    inference_config,
                    additional_model_fields):
        """
        Sends messages to a model and streams the response.
        Args:
            bedrock_client: The Boto3 Bedrock runtime client.
            model_id (str): The model ID to use.
            messages (JSON) : The messages to send.
            system_prompts (JSON) : The system prompts to send.
            inference_config (JSON) : The inference configuration to use.
            additional_model_fields (JSON) : Additional model fields to use.
        Returns:
            Stream object for processing response chunks.
        """

        logger.info("Streaming messages with model %s", model_id)

        params = {
            'modelId': model_id,
            'messages': messages,
            'system': system_prompts,
            'inferenceConfig': inference_config,
            'additionalModelRequestFields': additional_model_fields,
        }
        
        response = bedrock_client.client.converse_stream(**params) 
        return response.get('stream')

    def parse_stream(self, stream, tracker):
        """
        Process streaming response from Bedrock.
        
        Args:
            stream: Bedrock stream response
            tracker: LaunchDarkly tracker for metrics
            
        Yields:
            Message chunks for streaming display
            
        Returns:
            Complete response text
        """
        
        full_response = ""
        metric_response = {}
        metric_response["$metadata"] = {
            "httpStatusCode" : 200
        }
        
        # Add timing metrics and save to session
        start_time = time.time()
        st.session_state['start_time'] = start_time
        first_token_time = None
        
        for event in stream:
            if 'messageStart' in event:
                logger.info(f"Role: {event['messageStart']['role']}")

            if 'contentBlockDelta' in event:            
                message = event['contentBlockDelta']['delta']['text']
                # Record time of first token if not already set
                if first_token_time is None:
                    first_token_time = time.time()
                    time_to_first_token = (first_token_time - st.session_state['start_time']) * 1000
                    logger.info(f"Time to first token: {time_to_first_token} ms")
                    
                    
                    # Add to metrics
                    if "metrics" not in metric_response:
                        metric_response["metrics"] = {}
                    metric_response["metrics"]["timeToFirstToken"] = time_to_first_token
                
                full_response += message
                yield message # return output so chat can render it immediately

            if 'messageStop' in event:
                logger.info(f"Stop reason: {event['messageStop']['stopReason']}")

            if 'metadata' in event:
                metadata = event['metadata']
                if 'usage' in metadata:
                    logger.info("Token usage")
                    logger.info(f"Input tokens: {metadata['usage']['inputTokens']}")
                    logger.info(f":Output tokens: {metadata['usage']['outputTokens']}")
                    logger.info(f":Total tokens: {metadata['usage']['totalTokens']}")
                    metric_response["usage"] = metadata['usage']
                if 'metrics' in event['metadata']:
                    logger.info(f"Latency (Total Time for Response): {metadata['metrics']['latencyMs']} milliseconds")
                    if "metrics" not in metric_response:
                        metric_response["metrics"] = {}
                    metric_response["metrics"]["latencyMs"] = metadata['metrics']['latencyMs']
        
        # Add assistant response to chat history
        logger.info(f"Full response: {full_response}")
        
        # Send metrics to tracker
        if tracker:
            #Track AWS converse metrics
            tracker.track_bedrock_converse_metrics(metric_response)
            
            #Track success response
            tracker.track_success()
            
            #Track AI metrics individually to LaunchDarkly (eg. time to first token)
            tracker.track_time_to_first_token(time_to_first_token)                 
            
        # Store metrics in session state for display
        st.session_state['LDtracker'] = metric_response            
                
        return full_response

def create_bedrock_message(message_history: List[Dict[str, str]], current_prompt: str) -> List[Dict[str, Any]]:
    """
    Create a message array for Bedrock API that includes conversation history.
    
    Args:
        message_history: Previous messages in the conversation
        current_prompt: The current user input text
        
    Returns:
        Message array formatted for Bedrock
    """
    # Convert the conversation history to Bedrock format
    bedrock_messages = []
    
    # Add historical messages first (limited to last 10 to avoid context length issues)
    for msg in message_history[-10:]:
        role = "user" if msg["role"] == "Human" else "assistant"
        bedrock_messages.append({
            "role": role,
            "content": [{"text": msg["content"]}]
        })
    
    # Add the current user message
    bedrock_messages.append({
        "role": "user",
        "content": [{"text": current_prompt}]
    })
    
    return bedrock_messages

#######################################################################
###                                                                 ###
###                  SECTION 3: MAIN APPLICATION                    ###
###                                                                 ###
#######################################################################

def load_environment():
    """Load environment variables securely and validate them."""
    # Load from .env file
    dotenv.load_dotenv()
    
    # Check required variables
    required_vars = {
        "LD_SERVER_KEY": "Your LaunchDarkly SDK key",
        "AWS_REGION": "AWS region for Bedrock (e.g., us-west-2)",
    }
    
    missing = []
    for var, description in required_vars.items():
        if not os.getenv(var):
            missing.append(f"{var} ({description})")
    
    if missing:
        logger.error(f"Missing required environment variables: {', '.join(missing)}")
        return False
    
    # Sanitize and validate values
    aws_region = os.getenv("AWS_REGION")
    valid_regions = ["us-east-1", "us-west-2", "eu-central-1"]  # Example regions
    if aws_region not in valid_regions:
        logger.warning(f"AWS_REGION '{aws_region}' may not be valid for Bedrock")
    
    return True

def main():
    """Main application entry point."""
    try:
        # First validate environment
        if not load_environment():
            st.error("Environment configuration error. Check logs for details.")
            st.info("Make sure you have a properly configured .env file")
            return
        
        # Initialize session to store metrics
        if 'LDtracker' not in st.session_state:
            logger.info("Initializing session state")
            st.session_state['LDtracker'] = {}
    
        # Initialize LaunchDarkly client
        ld_client = LaunchDarklyClient(
            server_key=os.getenv("LD_SERVER_KEY"),
            ai_config_id=os.getenv("LD_AI_CONFIG_ID", default="test1")
        )
    
        # Display the logo
        st.logo(MAIN_LOGO)
        
        # Set up the sidebar for user profile configuration
        with st.sidebar:
            # Create variables to store form values
            if 'saved_user_info' not in st.session_state:
                st.session_state.saved_user_info = {
                    "name": "John Doe",
                    "age": 35,
                    "state": "California",
                    "claims_number": 2,
                    "policy_type": "Auto",
                    "coverage_level": "Premium",
                    "deductible": 500,
                    "premium": 1200,
                    "policy_start": "2023-01-01",
                    "policy_end": "2024-01-01"
                }
            
            # Display current values from saved state
            st.subheader("Current Profile:", divider="rainbow")
            st.write(f"Name: {st.session_state.saved_user_info['name']}")
            st.write(f"Age: {st.session_state.saved_user_info['age']}")
            st.write(f"State: {st.session_state.saved_user_info['state']}")
            st.write(f"Claims Number: {st.session_state.saved_user_info['claims_number']}")
            st.write(f"Policy Type: {st.session_state.saved_user_info['policy_type']}")
            st.write(f"Coverage Level: {st.session_state.saved_user_info['coverage_level']}")
            st.write(f"Deductible: ${st.session_state.saved_user_info['deductible']}")
            st.write(f"Premium: ${st.session_state.saved_user_info['premium']}")
            st.write(f"Policy Start: {st.session_state.saved_user_info['policy_start']}")
            st.write(f"Policy End: {st.session_state.saved_user_info['policy_end']}")
            
            # Create function to update profile
            def update_profile():
                st.session_state.saved_user_info = {
                    "name": st.session_state.temp_name,
                    "age": st.session_state.temp_age,
                    "state": st.session_state.temp_state,
                    "claims_number": st.session_state.temp_claims_number,
                    "policy_type": st.session_state.temp_policy_type,
                    "coverage_level": st.session_state.temp_coverage_level,
                    "deductible": st.session_state.temp_deductible,
                    "premium": st.session_state.temp_premium,
                    "policy_start": st.session_state.temp_policy_start,
                    "policy_end": st.session_state.temp_policy_end
                }
                # Set flag to display success message after rerun
                st.session_state.show_success = True
                
            # Show success message if flag is set
            if st.session_state.get('show_success', False):
                st.success("Profile updated successfully!")
                # Reset flag so message doesn't show on subsequent reruns
                st.session_state.show_success = False
            
            st.subheader("Update Profile", divider="rainbow")
            # Form inputs using session state keys directly for data binding
            with st.form(key="user_profile_form"):
                st.text_input("Name", value=st.session_state.saved_user_info["name"], key="temp_name")
                st.number_input("Age", 0, 100, value=st.session_state.saved_user_info["age"], key="temp_age")
                st.text_input("State", value=st.session_state.saved_user_info["state"], key="temp_state")
                st.number_input("Claims Number", 0, 50, value=st.session_state.saved_user_info["claims_number"], key="temp_claims_number")
                st.selectbox("Policy Type", options=["Auto", "Home", "Life", "Health"], 
                           index=["Auto", "Home", "Life", "Health"].index(st.session_state.saved_user_info["policy_type"]), 
                           key="temp_policy_type")
                st.selectbox("Coverage Level", options=["Basic", "Standard", "Premium", "Elite"], 
                           index=["Basic", "Standard", "Premium", "Elite"].index(st.session_state.saved_user_info["coverage_level"]), 
                           key="temp_coverage_level")
                st.number_input("Deductible Amount ($)", 0, 10000, value=st.session_state.saved_user_info["deductible"], key="temp_deductible")
                st.number_input("Premium ($)", 0, 10000, value=st.session_state.saved_user_info["premium"], key="temp_premium")
                st.text_input("Policy Start Date (YYYY-MM-DD)", value=st.session_state.saved_user_info["policy_start"], key="temp_policy_start")
                st.text_input("Policy End Date (YYYY-MM-DD)", value=st.session_state.saved_user_info["policy_end"], key="temp_policy_end")
                
                # Submit button with on_click action that triggers after form validation
                submit_button = st.form_submit_button(label="Save Changes", on_click=update_profile)
            
                if submit_button:
                    new_context = Context.builder("unique-user") \
                        .set("name", st.session_state.temp_name) \
                        .set("age", st.session_state.temp_age) \
                        .set("state", st.session_state.temp_state) \
                        .set("claims_number", st.session_state.temp_claims_number) \
                        .set("policy_type", st.session_state.temp_policy_type) \
                        .set("coverage_level", st.session_state.temp_coverage_level) \
                        .set("deductible", st.session_state.temp_deductible) \
                        .set("premium", st.session_state.temp_premium) \
                        .set("policy_start", st.session_state.temp_policy_start) \
                        .set("policy_end", st.session_state.temp_policy_end) \
                        .build()
                    ldclient.get().identify(new_context)

            st.divider()

            # Get conversation history for context
            conversation_history = get_history()
            
            # LaunchDarkly AI Config - use saved values instead of form values
            context = Context.builder(st.session_state.saved_user_info["name"]) \
                .set("name", st.session_state.saved_user_info["name"]) \
                .set("age", st.session_state.saved_user_info["age"]) \
                .set("state", st.session_state.saved_user_info["state"]) \
                .set("claims_number", st.session_state.saved_user_info["claims_number"]) \
                .set("policy_type", st.session_state.saved_user_info["policy_type"]) \
                .set("coverage_level", st.session_state.saved_user_info["coverage_level"]) \
                .set("deductible", st.session_state.saved_user_info["deductible"]) \
                .set("premium", st.session_state.saved_user_info["premium"]) \
                .set("policy_start", st.session_state.saved_user_info["policy_start"]) \
                .set("policy_end", st.session_state.saved_user_info["policy_end"]) \
                .build()

            variables = { 
                "user_input": st.session_state.get("user_input", ""),
                "conversation_history": conversation_history
            }
            
            # Get AI configuration from LaunchDarkly
            config, tracker = ld_client.get_ai_config(context, variables)
            
            # Extract configuration for Bedrock
            # System prompt must be a dictionary with 'text' key for Bedrock Converse API
            system_prompts = [{"text": config.messages[0].content}]
            
            # Map model parameters to Bedrock expected format
            model_params = config.model._parameters
            
            # Handle different parameter access methods based on AIConfig structure
            if isinstance(model_params, dict):
                # Use the dict directly
                params_dict = model_params
            else:
                # Try to access as object attributes or use defaults
                params_dict = {}
                # Get temperature with fallback
                if hasattr(model_params, "temperature"):
                    params_dict["temperature"] = model_params.temperature
                # Get top_p with fallback
                if hasattr(model_params, "top_p"):
                    params_dict["top_p"] = model_params.top_p
                # Get max_tokens with fallback
                if hasattr(model_params, "max_tokens"):
                    params_dict["max_tokens"] = model_params.max_tokens
            
            # Create inference config with proper Bedrock parameter names
            inference_config = {
                # Convert snake_case to camelCase for Bedrock API
                "temperature": params_dict.get("temperature"),
                "topP": params_dict.get("top_p"),
                "maxTokens": params_dict.get("max_tokens")
            }
            # Remove any params that are None to avoid validation errors
            inference_config = {k: v for k, v in inference_config.items() if v is not None}
            
            # Additional model fields in correct format for Bedrock
            additional_model_fields = {}
            
            # Display configuration details in expandable sections
            with st.expander("System prompt", expanded=False):
                st.text("This is the system prompt that is dynamically rendered based on user profile:")
                st.json(json.dumps(system_prompts, indent=2))
    
            with st.expander("LaunchDarkly AI config data", expanded=False):
                st.text("This is the context stored in AI Config for this particular user:")
                st.text("Context Info:")
                st.json(context.to_json_string())
                st.text("This is the LLM configuration that is dynamically rendered based on user profile:")
                st.text("Model Info:")
                st.json(config.model.to_dict())
            
            with st.expander("LaunchDarkly AI config tracker", expanded=False):
                st.text("This is the tracker that is dynamically sent as telemetry to LaunchDarkly:")
                if 'LDtracker' in st.session_state:
                    st.json(st.session_state['LDtracker'])
                else:
                    st.code("No tracker yet")

        # Initialize bedrock client and pick model from AIConfig
        bedrock_client = get_bedrock_client()
        model_id = config.model.name

        # Initialize welcome message
        st.subheader(f"Hello {st.session_state.saved_user_info['name']}, I am your Insurance Assistant!")
        welcome_message = get_welcome_message()
        with st.chat_message('assistant'):
            st.markdown(welcome_message)
            
        # Add conversation context display
        with st.expander("Conversation Context", expanded=False):
            st.text("This is what the AI model remembers from your conversation:")
            if "messages" in st.session_state and st.session_state.messages:
                for msg in st.session_state.messages[-5:]:
                    st.markdown(f"**{msg['role']}**: {msg['content'][:100]}..." if len(msg['content']) > 100 else f"**{msg['role']}**: {msg['content']}")
            else:
                st.info("No conversation history yet")
            
        # Initialize chat history
        if "messages" not in st.session_state:
            st.session_state.messages = []
    
        # Display chat messages from history on app rerun
        for i, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                
                # Show feedback buttons for assistant messages
                if message["role"] == "Assistant":
                    # Create a unique key for this message's feedback
                    message_key = f"feedback_{i}"
                    
                    st.write("Was this response helpful?")
                    col1, col2, col3 = st.columns([1, 1, 3])
                    
                    # Check if feedback was already submitted for this message
                    feedback_submitted = st.session_state.get('feedback_status', {}).get(i, {}).get('submitted', False)
                    
                    if not feedback_submitted:
                        with col1:
                            if st.button("👍 Yes", key=f"thumbs_up_{i}"):
                                ld_send_feedback(tracker, True, i)
                                st.rerun()
                        
                        with col2:
                            if st.button("👎 No", key=f"thumbs_down_{i}"):
                                ld_send_feedback(tracker, False, i)
                                st.rerun()
                    else:
                        # Show confirmation message after feedback is submitted
                        feedback_value = st.session_state.get('feedback_status', {}).get(i, {}).get('value', None)
                        if feedback_value is not None:
                            if feedback_value:
                                st.success("Thank you for your positive feedback!")
                            else:
                                st.info("Thank you for your feedback! Your input helps us improve our responses.")
    
        # React to user input
        if prompt := st.chat_input("How can I assist you with your insurance needs today?"):        
            # Store the user input in session state for LaunchDarkly variables
            st.session_state.user_input = prompt
            
            # Display user message in chat message container
            with st.chat_message("Human"):
                st.markdown(prompt)
    
            # Add user message to chat history
            st.session_state.messages.append({"role": "Human", "content": prompt})
    
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                # Create message for Bedrock API - now including conversation history
                messages = create_bedrock_message(st.session_state.messages[:-1], prompt)
    
                try:
                    # Show loading status while generating response
                    message_placeholder = st.empty()
                    message_placeholder.info("Generating response...")
                    
                    # Update variables with current input for LaunchDarkly
                    variables = { 
                        "user_input": prompt,
                        "conversation_history": get_history()
                    }
                    
                    # Refresh the AI config with new conversation context
                    config, tracker = ld_client.get_ai_config(context, variables)
                    
                    # Update system prompt and model parameters in case they changed based on context
                    system_prompts = [{"text": config.messages[0].content}]
                    
                    stream = bedrock_client.stream_conversation(bedrock_client, model_id, messages, system_prompts, inference_config, additional_model_fields)
                    if stream:
                        full_response = st.write_stream(bedrock_client.parse_stream(stream, tracker))
                        # Clear the "Generating..." message
                        message_placeholder.empty()
                        # Display the complete response
                        st.markdown(full_response)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append(
                        {"role": "Assistant", "content": full_response}
                    )
                    
                    # Render update
                    st.rerun()
                        
                except ClientError as err:
                    message = err.response['Error']['Message']
                    logger.error(f"A client error occurred: {message}")
                    st.error(f"Error: {message}")
                except Exception as e:
                    logger.error(f"An error occurred: {str(e)}")
                    st.error(f"Error: {str(e)}")
                else:
                    logger.info(f"Finished streaming messages with model {model_id}.")
    except Exception as e:
        logger.exception("Unexpected error in main application")
        st.error(f"An unexpected error occurred: {str(e)}")


#######################################################################
###                                                                 ###
###                 SECTION 4: HELPER FUNCTIONS                     ###
###                                                                 ###
#######################################################################

def ld_send_feedback(tracker, is_positive, message_index):
    """
    Send feedback to LaunchDarkly.
    
    Args:
        tracker: LaunchDarkly tracker object
        is_positive: Whether feedback is positive
        message_index: Index of the message receiving feedback
    """
    try:
        # Get the LaunchDarkly client instance
        ld_client = LaunchDarklyClient(
            server_key=os.getenv("LD_SERVER_KEY"),
            ai_config_id=os.getenv("LD_AI_CONFIG_ID")
        )
        ld_client.send_feedback(tracker, is_positive)
        
        # Log the feedback for debugging
        logger.info(f"Feedback sent to LaunchDarkly: {'positive' if is_positive else 'negative'}")
        
        # Store the feedback in session state for display
        if 'feedback_status' not in st.session_state:
            st.session_state['feedback_status'] = {}
        st.session_state['feedback_status'][message_index] = {
            'submitted': True,
            'value': is_positive
        }
    except Exception as e:
        logger.error(f"Error sending feedback: {str(e)}")

@st.cache_data
def get_welcome_message() -> str:
    """Get a random welcome message for the insurance AI chat interface."""
    return random.choice(
        [
            "Welcome to Insurance Advisor! I'm here to help with your insurance questions and needs. How may I assist you today?",
            "Hello! I'm your insurance assistant. What questions do you have about your policy or coverage?",
            "Greetings! I'm your insurance advisor. Feel free to ask me anything about your policy, claims, or coverage options.",
            "Welcome to your personal insurance assistant. How may I help with your insurance needs today?",
            "Hi there! I'm ready to provide insurance information and guidance. What would you like to know about your policy?"
        ]
    )

@st.cache_resource
def get_bedrock_client():
    """Get cached Bedrock client instance."""
    return BedrockClient(region_name=os.getenv("AWS_REGION"))

def get_history() -> str:
    """Get chat history as formatted text for context."""
    if not hasattr(st.session_state, 'messages') or not st.session_state.messages:
        return ""
        
    # Format history with clear role indicators and optimize for context
    formatted_history = []
    for record in st.session_state.messages[-6:]:  # Limit to last 6 messages to avoid context length issues
        role = "User" if record['role'] == "Human" else "Assistant"
        formatted_history.append(f"{role}: {record['content']}")
    
    return '\n\n'.join(formatted_history)

def setup_logging():
    """Configure logging to write to both console and file."""
    log_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        handlers=[
            logging.FileHandler("insurance_chatbot.log"),
            logging.StreamHandler()
        ]
    )

# Cache the user context to avoid unnecessary recreation
@st.cache_resource(ttl=300)  # Cache for 5 minutes
def get_user_context(user_info):
    """Create and cache the user context to avoid recreating it frequently."""
    return Context.builder(user_info["name"]) \
        .set("name", user_info["name"]) \
        .set("age", user_info["age"]) \
        .set("state", user_info["state"]) \
        .set("claims_number", user_info["claims_number"]) \
        .set("policy_type", user_info["policy_type"]) \
        .set("coverage_level", user_info["coverage_level"]) \
        .set("deductible", user_info["deductible"]) \
        .set("premium", user_info["premium"]) \
        .set("policy_start", user_info["policy_start"]) \
        .set("policy_end", user_info["policy_end"]) \
        .build()

# Run the app when this file is executed directly
if __name__ == "__main__":
    setup_logging()
    main()