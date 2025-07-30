"""
Agent Factory Module
Provides standardized agent creation with error handling and configuration management
"""

import logging
from typing import Dict, Optional
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from config.app_config import AppConfig

logger = logging.getLogger(__name__)

class AgentFactory:
    """Factory for creating and managing AI agents with standardized configuration"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        """
        Initialize agent factory
        
        Args:
            config: Application configuration (will use AppConfig if not provided)
        """
        self.config = config or AppConfig
        self._client: Optional[OpenAIChatCompletionClient] = None
        self._agents: Dict[str, AssistantAgent] = {}
    
    def _initialize_client(self) -> OpenAIChatCompletionClient:
        """Initialize OpenAI client with Azure configuration"""
        if self._client is not None:
            return self._client
            
        try:
            # Ensure configuration is initialized
            if not self.config._initialized:
                raise RuntimeError("AppConfig not initialized. Call AppConfig.initialize() first.")
            
            # Validate required configuration
            self.config.validate_required()
            
            # Get formatted endpoint URL
            full_endpoint = self.config.get_azure_endpoint_url()
            
            # Log configuration initialization
            logger.info("Initializing Azure OpenAI client with configured settings")
            
            # Create the client with explicit Azure configuration
            self._client = OpenAIChatCompletionClient(
                model=self.config.AZURE_OPENAI_DEPLOYMENT_NAME,
                api_key=self.config.AZURE_OPENAI_API_KEY,
                api_base=full_endpoint,
                api_version=self.config.AZURE_OPENAI_API_VERSION,
                api_type="azure",
                deployment_id=self.config.AZURE_OPENAI_DEPLOYMENT_NAME,
                base_url=full_endpoint,
                default_headers={
                    "api-key": self.config.AZURE_OPENAI_API_KEY,
                    "Content-Type": "application/json"
                },
                default_query={
                    "api-version": self.config.AZURE_OPENAI_API_VERSION
                }
            )
            
            logger.info("Successfully initialized Azure OpenAI client")
            return self._client
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure OpenAI client: {str(e)}")
            raise RuntimeError(f"Azure OpenAI configuration error: {str(e)}")
    
    def create_agent(self, agent_type: str, system_message: str, force_recreate: bool = False) -> AssistantAgent:
        """
        Create an agent with standardized error handling
        
        Args:
            agent_type: Type of agent (e.g., 'context', 'plumbing')
            system_message: System message for the agent
            force_recreate: Force recreation even if agent already exists
            
        Returns:
            Configured AssistantAgent instance
        """
        agent_name = f"{agent_type}_agent"
        
        # Return existing agent if available and not forcing recreation
        if not force_recreate and agent_name in self._agents:
            logger.debug(f"Returning existing {agent_type} agent")
            return self._agents[agent_name]
        
        try:
            # Initialize client if needed
            client = self._initialize_client()
            
            # Create the agent
            agent = AssistantAgent(
                name=agent_name,
                model_client=client,
                system_message=system_message
            )
            
            # Cache the agent
            self._agents[agent_name] = agent
            
            logger.info(f"Successfully created {agent_type} agent")
            return agent
            
        except Exception as e:
            logger.error(f"Failed to create {agent_type} agent: {str(e)}")
            raise RuntimeError(f"{agent_type.title()} agent creation error: {str(e)}")
    
    def get_agent(self, agent_type: str) -> Optional[AssistantAgent]:
        """
        Get an existing agent by type
        
        Args:
            agent_type: Type of agent to retrieve
            
        Returns:
            Agent instance if it exists, None otherwise
        """
        agent_name = f"{agent_type}_agent"
        return self._agents.get(agent_name)
    
    def create_standard_agents(self, system_messages: Dict[str, str]) -> Dict[str, AssistantAgent]:
        """
        Create all standard agents for the analysis pipeline
        
        Args:
            system_messages: Dictionary mapping agent types to their system messages
            
        Returns:
            Dictionary of created agents
        """
        agents = {}
        
        for agent_type, system_message in system_messages.items():
            try:
                agent = self.create_agent(agent_type, system_message)
                agents[agent_type] = agent
            except Exception as e:
                logger.error(f"Failed to create {agent_type} agent during batch creation: {str(e)}")
                # Continue creating other agents even if one fails
                continue
        
        logger.info(f"Successfully created {len(agents)}/{len(system_messages)} agents")
        return agents
    
    def reset_agents(self) -> None:
        """Reset all cached agents (forces recreation on next request)"""
        self._agents.clear()
        logger.info("All cached agents have been reset")
    
    def get_client(self) -> OpenAIChatCompletionClient:
        """Get the OpenAI client (initializes if needed)"""
        return self._initialize_client()
    
    def get_agent_summary(self) -> Dict[str, str]:
        """Get summary of created agents"""
        return {
            "total_agents": len(self._agents),
            "agent_types": list(self._agents.keys()),
            "client_initialized": self._client is not None
        } 