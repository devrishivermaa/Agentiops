# utils/mongo_helper.py
"""
MongoDB connection helper with proper SSL/TLS configuration for MongoDB Atlas.
"""

import os
import ssl
import certifi
from typing import Optional, Tuple
from pymongo import MongoClient
from pymongo.database import Database
from pymongo.collection import Collection
from utils.logger import get_logger

logger = get_logger("MongoHelper")


def get_mongo_client(uri: str = None, timeout_ms: int = 30000) -> Optional[MongoClient]:
    """
    Create a MongoDB client with proper SSL/TLS configuration for MongoDB Atlas.
    
    Args:
        uri: MongoDB connection URI. If None, uses MONGO_URI env var.
        timeout_ms: Server selection timeout in milliseconds.
        
    Returns:
        MongoClient instance or None if connection fails.
    """
    uri = uri or os.getenv("MONGO_URI")
    
    if not uri:
        logger.warning("MONGO_URI not provided or not set in environment")
        return None
    
    # Check if it's a MongoDB Atlas URI (mongodb+srv://)
    is_atlas = uri.startswith("mongodb+srv://")
    
    try:
        if is_atlas:
            # MongoDB Atlas with proper SSL using certifi
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=timeout_ms,
                connectTimeoutMS=timeout_ms,
                socketTimeoutMS=timeout_ms,
                tlsCAFile=certifi.where(),  # Use certifi's CA bundle for SSL
                retryWrites=True,
                w="majority"
            )
        else:
            # Regular MongoDB connection
            client = MongoClient(
                uri,
                serverSelectionTimeoutMS=timeout_ms,
                connectTimeoutMS=timeout_ms,
                socketTimeoutMS=timeout_ms,
            )
        
        # Test the connection
        client.admin.command('ping')
        logger.info("Successfully connected to MongoDB")
        return client
        
    except Exception as e:
        logger.warning(f"MongoDB connection failed with standard SSL: {e}")
        
        # Fallback: try with different SSL settings for Atlas
        if is_atlas:
            try:
                # Try with explicit TLS settings
                client = MongoClient(
                    uri,
                    serverSelectionTimeoutMS=timeout_ms,
                    connectTimeoutMS=timeout_ms,
                    socketTimeoutMS=timeout_ms,
                    tls=True,
                    tlsAllowInvalidCertificates=True,  # Last resort
                    retryWrites=True,
                    w="majority"
                )
                
                client.admin.command('ping')
                logger.info("Connected to MongoDB Atlas with relaxed SSL")
                return client
                
            except Exception as e2:
                logger.error(f"MongoDB Atlas connection failed completely: {e2}")
                return None
        else:
            logger.error(f"MongoDB connection failed: {e}")
            return None


def get_mongo_collection(
    collection_name: str,
    db_name: str = None,
    uri: str = None
) -> Tuple[Optional[MongoClient], Optional[Collection]]:
    """
    Get a MongoDB collection with proper error handling.
    
    Args:
        collection_name: Name of the collection.
        db_name: Database name. If None, uses MONGO_DB env var.
        uri: MongoDB URI. If None, uses MONGO_URI env var.
        
    Returns:
        Tuple of (MongoClient, Collection) or (None, None) if connection fails.
    """
    db_name = db_name or os.getenv("MONGO_DB")
    
    if not db_name:
        logger.warning("MONGO_DB not provided or not set in environment")
        return None, None
    
    client = get_mongo_client(uri)
    
    if client is None:
        return None, None
    
    try:
        db = client[db_name]
        collection = db[collection_name]
        return client, collection
    except Exception as e:
        logger.error(f"Failed to get collection {collection_name}: {e}")
        return None, None


def safe_mongo_insert(collection: Optional[Collection], document: dict) -> bool:
    """
    Safely insert a document into MongoDB collection.
    
    Args:
        collection: MongoDB collection (can be None).
        document: Document to insert.
        
    Returns:
        True if insert succeeded, False otherwise.
    """
    if collection is None:
        logger.debug("MongoDB collection not available, skipping insert")
        return False
    
    try:
        result = collection.insert_one(document)
        logger.debug(f"Document inserted with ID: {result.inserted_id}")
        return True
    except Exception as e:
        logger.error(f"MongoDB insert failed: {e}")
        return False


def safe_mongo_find_one(collection: Optional[Collection], query: dict) -> Optional[dict]:
    """
    Safely find a document in MongoDB collection.
    
    Args:
        collection: MongoDB collection (can be None).
        query: Query to execute.
        
    Returns:
        Found document or None.
    """
    if collection is None:
        return None
    
    try:
        return collection.find_one(query)
    except Exception as e:
        logger.error(f"MongoDB find failed: {e}")
        return None
