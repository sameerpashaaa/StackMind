#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union

from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from utils.helpers import get_current_datetime

logger = logging.getLogger(__name__)

class MemorySystem:
   # Valid memory item types
    MEMORY_TYPES = ["input", "solution", "feedback", "preference", "context"]
    
    def __init__(self, settings):
        """
        Initialize the Memory System.
        
        Args:
            settings: Application settings object
        """
        self.settings = settings
        self.memory_settings = settings.get_section("memory")
        
        # Initialize session memory
        self.session_memory = {
            "inputs": [],
            "solutions": [],
            "feedback": [],
            "preferences": [],
            "context": []
        }
        
        # Initialize long-term memory if persistence is enabled
        self.long_term_memory = None
        if self.memory_settings.get("persistence", True):
            self._initialize_long_term_memory()
        
        logger.info("Memory system initialized")
    
    def _initialize_long_term_memory(self):
        """
        Initialize the long-term memory vector store.
        """
        try:
            # Create storage directory if it doesn't exist
            storage_path = self.memory_settings.get("storage_path", "./data/memory")
            os.makedirs(storage_path, exist_ok=True)
            
            # Initialize embeddings
            embeddings = OpenAIEmbeddings()
            
            # Initialize vector store
            self.long_term_memory = Chroma(
                collection_name="problem_solver_memory",
                embedding_function=embeddings,
                persist_directory=storage_path
            )
            
            logger.info(f"Long-term memory initialized at {storage_path}")
        except Exception as e:
            logger.error(f"Failed to initialize long-term memory: {e}")
            self.long_term_memory = None
    
    def add(self, item: Dict[str, Any], item_type: str) -> str:
        """
        Add an item to memory.
        
        Args:
            item: The item to add to memory
            item_type: Type of memory item (input, solution, feedback, preference, context)
            
        Returns:
            str: ID of the added memory item
        """
        # Validate item type
        if item_type not in self.MEMORY_TYPES:
            logger.warning(f"Invalid memory item type: {item_type}. Using 'context'.")
            item_type = "context"
        
        # Add metadata if not present
        if "id" not in item:
            item["id"] = str(uuid.uuid4())
        if "timestamp" not in item:
            item["timestamp"] = get_current_datetime()
        if "type" not in item:
            item["type"] = item_type
        
        # Add to session memory
        session_key = f"{item_type}s" if item_type != "feedback" else "feedback"
        self.session_memory[session_key].append(item)
        
        # Trim session memory if it exceeds the maximum size
        max_history = self.memory_settings.get("max_history", 100)
        if len(self.session_memory[session_key]) > max_history:
            self.session_memory[session_key] = self.session_memory[session_key][-max_history:]
        
        # Add to long-term memory if enabled
        if self.long_term_memory is not None:
            try:
                # Convert item to string for storage
                item_str = json.dumps(item)
                
                # Create document for vector store
                doc = Document(
                    page_content=item_str,
                    metadata={
                        "id": item["id"],
                        "type": item_type,
                        "timestamp": item["timestamp"]
                    }
                )
                
                # Add to vector store
                self.long_term_memory.add_documents([doc])
                
                logger.debug(f"Added item to long-term memory: {item['id']}")
            except Exception as e:
                logger.error(f"Failed to add item to long-term memory: {e}")
        
        logger.debug(f"Added {item_type} to memory: {item['id']}")
        return item["id"]
    
    def get(self, item_id: str) -> Optional[Dict[str, Any]]:
        """
        Get a specific memory item by ID.
        
        Args:
            item_id: ID of the memory item to retrieve
            
        Returns:
            Optional[Dict[str, Any]]: The memory item or None if not found
        """
        # Check session memory first
        for memory_type in self.session_memory:
            for item in self.session_memory[memory_type]:
                if item.get("id") == item_id:
                    return item
        
        # Check long-term memory if enabled
        if self.long_term_memory is not None:
            try:
                # Search for the item by ID
                results = self.long_term_memory.similarity_search(
                    query=item_id,
                    k=1,
                    filter={"id": item_id}
                )
                
                if results:
                    # Parse the item from the document content
                    item_str = results[0].page_content
                    return json.loads(item_str)
            except Exception as e:
                logger.error(f"Failed to retrieve item from long-term memory: {e}")
        
        logger.warning(f"Memory item not found: {item_id}")
        return None
    
    def search(self, query: str, k: int = 5, item_type: Optional[str] = None) -> List[Dict[str, Any]]:
        results = []
        
        # Search long-term memory if enabled
        if self.long_term_memory is not None:
            try:
                # Prepare filter
                filter_dict = {}
                if item_type:
                    filter_dict["type"] = item_type
                
                # Perform similarity search
                docs = self.long_term_memory.similarity_search(
                    query=query,
                    k=k,
                    filter=filter_dict if filter_dict else None
                )
                
                # Parse results
                for doc in docs:
                    try:
                        item = json.loads(doc.page_content)
                        results.append(item)
                    except json.JSONDecodeError:
                        logger.warning(f"Failed to parse memory item: {doc.page_content[:100]}...")
            except Exception as e:
                logger.error(f"Failed to search long-term memory: {e}")
        
        # If no results from long-term memory or it's disabled, search session memory
        if not results:
            # Combine all session memory items
            all_items = []
            for memory_type, items in self.session_memory.items():
                if not item_type or memory_type.rstrip("s") == item_type:
                    all_items.extend(items)
            
            # Sort by recency (most recent first)
            all_items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            # Return the most recent items (simple fallback strategy)
            results = all_items[:k]
        
        logger.debug(f"Memory search for '{query}' returned {len(results)} results")
        return results
    
    def get_recent(self, item_type: Optional[str] = None, limit: int = 10) -> List[Dict[str, Any]]:
        if item_type:
            if item_type not in self.MEMORY_TYPES:
                logger.warning(f"Invalid memory item type: {item_type}")
                return []
            
            session_key = f"{item_type}s" if item_type != "feedback" else "feedback"
            items = self.session_memory[session_key][-limit:]
            return items
        else:
            # Combine all memory types
            all_items = []
            for memory_type, items in self.session_memory.items():
                all_items.extend(items)
            
            # Sort by timestamp (most recent first)
            all_items.sort(key=lambda x: x.get("timestamp", ""), reverse=True)
            
            return all_items[:limit]
    
    def get_user_preferences(self) -> Dict[str, Any]:
        
        preferences = {}
        
        # Collect all preference items
        preference_items = self.session_memory["preferences"]
        
        # Add preferences from long-term memory if enabled
        if self.long_term_memory is not None:
            try:
                docs = self.long_term_memory.similarity_search(
                    query="user preferences",
                    k=50,
                    filter={"type": "preference"}
                )
                
                for doc in docs:
                    try:
                        item = json.loads(doc.page_content)
                        preference_items.append(item)
                    except json.JSONDecodeError:
                        continue
            except Exception as e:
                logger.error(f"Failed to retrieve preferences from long-term memory: {e}")
        
        # Aggregate preferences (more recent preferences override older ones)
        for item in sorted(preference_items, key=lambda x: x.get("timestamp", "")):
            prefs = item.get("preferences", {})
            preferences.update(prefs)
        
        return preferences
    
    def add_user_preference(self, preference_key: str, preference_value: Any) -> None:
        
        # Create preference item
        preference_item = {
            "id": str(uuid.uuid4()),
            "timestamp": get_current_datetime(),
            "type": "preference",
            "preferences": {preference_key: preference_value}
        }
        
        # Add to memory
        self.add(preference_item, "preference")
        
        logger.debug(f"Added user preference: {preference_key}={preference_value}")
    
    def clear_session(self) -> None:
        """
        Clear the session memory.
        """
        self.session_memory = {
            "inputs": [],
            "solutions": [],
            "feedback": [],
            "preferences": [],
            "context": []
        }
        
        logger.info("Session memory cleared")
    
    def prune_long_term_memory(self, days: Optional[int] = None) -> int:
        
        if self.long_term_memory is None:
            return 0
        
        try:
            # Get retention period from settings if not specified
            if days is None:
                days = self.memory_settings.get("data_retention_days", 30)
            
            # Calculate cutoff date
            cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
            
            # Get all documents
            all_docs = self.long_term_memory.get()
            
            # Identify documents to delete
            docs_to_delete = []
            for doc_id, doc in all_docs.items():
                timestamp = doc.metadata.get("timestamp", "")
                if timestamp and timestamp < cutoff_date:
                    docs_to_delete.append(doc_id)
            
            # Delete old documents
            if docs_to_delete:
                self.long_term_memory.delete(docs_to_delete)
            
            logger.info(f"Pruned {len(docs_to_delete)} items from long-term memory")
            return len(docs_to_delete)
        except Exception as e:
            logger.error(f"Failed to prune long-term memory: {e}")
            return 0