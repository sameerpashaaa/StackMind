#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Visualization Utilities

This module provides functions for visualizing problem-solving processes,
solution plans, and other data structures to enhance understanding and interaction.
"""

import logging
from typing import Dict, List, Any, Optional, Union, Tuple
import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

def create_solution_tree(steps: List[Dict], title: str = "Solution Tree") -> str:
    """
    Create a visual representation of a solution tree using NetworkX and Matplotlib.
    
    Args:
        steps (List[Dict]): List of solution steps with parent-child relationships
        title (str): Title of the visualization
        
    Returns:
        str: Path to the saved visualization image
    """
    try:
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for step in steps:
            step_id = step.get('id')
            parent_id = step.get('parent_id')
            label = step.get('label', f"Step {step_id}")
            
            G.add_node(step_id, label=label)
            
            if parent_id is not None:
                G.add_edge(parent_id, step_id)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color="skyblue")
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
        
        # Draw labels
        node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        plt.title(title)
        plt.axis("off")
        
        # Save the figure
        output_path = "solution_tree.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Solution tree visualization saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Failed to create solution tree visualization: {e}")
        return ""

def create_interactive_graph(nodes: List[Dict], edges: List[Dict], title: str = "Interactive Graph") -> str:
    """
    Create an interactive network visualization using PyVis.
    
    Args:
        nodes (List[Dict]): List of node dictionaries with 'id' and 'label' keys
        edges (List[Dict]): List of edge dictionaries with 'from', 'to', and optional 'label' keys
        title (str): Title of the visualization
        
    Returns:
        str: Path to the saved HTML file
    """
    try:
        # Create a network
        net = Network(height="750px", width="100%", directed=True, notebook=False)
        net.set_options("""
        var options = {
            "nodes": {
                "shape": "dot",
                "size": 20,
                "font": {
                    "size": 14,
                    "face": "Tahoma"
                }
            },
            "edges": {
                "color": {
                    "inherit": true
                },
                "smooth": {
                    "enabled": false
                }
            },
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "timestep": 0.35,
                "stabilization": {
                    "enabled": true,
                    "iterations": 1000
                }
            }
        }
        """)
        
        # Add nodes
        for node in nodes:
            net.add_node(node['id'], label=node['label'], title=node.get('title', node['label']))
        
        # Add edges
        for edge in edges:
            net.add_edge(
                edge['from'], 
                edge['to'], 
                label=edge.get('label', ''),
                title=edge.get('title', '')
            )
        
        # Save the visualization
        output_path = "interactive_graph.html"
        net.save_graph(output_path)
        
        logger.info(f"Interactive graph visualization saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Failed to create interactive graph visualization: {e}")
        return ""

def create_flowchart(steps: List[Dict], title: str = "Solution Flowchart") -> str:
    """
    Create a flowchart visualization of solution steps.
    
    Args:
        steps (List[Dict]): List of solution steps with 'id', 'label', and 'next' keys
        title (str): Title of the visualization
        
    Returns:
        str: Path to the saved visualization image
    """
    try:
        # Create a directed graph
        G = nx.DiGraph()
        
        # Add nodes and edges
        for step in steps:
            step_id = step.get('id')
            label = step.get('label', f"Step {step_id}")
            next_steps = step.get('next', [])
            
            G.add_node(step_id, label=label)
            
            for next_step in next_steps:
                G.add_edge(step_id, next_step)
        
        # Create the visualization
        plt.figure(figsize=(12, 8))
        pos = nx.spring_layout(G, seed=42)  # Positions for all nodes
        
        # Draw nodes with different colors based on node type
        node_colors = []
        for node in G.nodes():
            if G.out_degree(node) == 0:  # End nodes
                node_colors.append("lightgreen")
            elif G.in_degree(node) == 0:  # Start nodes
                node_colors.append("lightblue")
            else:  # Intermediate nodes
                node_colors.append("lightyellow")
        
        nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors)
        
        # Draw edges
        nx.draw_networkx_edges(G, pos, width=2, arrowsize=20)
        
        # Draw labels
        node_labels = {node: G.nodes[node]['label'] for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=10)
        
        plt.title(title)
        plt.axis("off")
        
        # Save the figure
        output_path = "solution_flowchart.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Flowchart visualization saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Failed to create flowchart visualization: {e}")
        return ""

def create_comparison_chart(data: List[Dict], x_key: str, y_key: str, title: str = "Comparison Chart") -> str:
    """
    Create a bar chart comparing different solutions or approaches.
    
    Args:
        data (List[Dict]): List of data points with x_key and y_key values
        x_key (str): Key for x-axis values
        y_key (str): Key for y-axis values
        title (str): Title of the visualization
        
    Returns:
        str: Path to the saved visualization image
    """
    try:
        # Extract data
        x_values = [item[x_key] for item in data]
        y_values = [item[y_key] for item in data]
        
        # Create the visualization
        plt.figure(figsize=(10, 6))
        plt.bar(x_values, y_values, color='skyblue')
        plt.xlabel(x_key.capitalize())
        plt.ylabel(y_key.capitalize())
        plt.title(title)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Save the figure
        output_path = "comparison_chart.png"
        plt.savefig(output_path, bbox_inches="tight")
        plt.close()
        
        logger.info(f"Comparison chart saved to {output_path}")
        return output_path
    
    except Exception as e:
        logger.error(f"Failed to create comparison chart: {e}")
        return ""