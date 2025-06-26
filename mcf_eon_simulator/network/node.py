#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Network Node Implementation for MCF EON
"""

from typing import List, Dict, Set
from dataclasses import dataclass, field

@dataclass
class NodeCapabilities:
    """Node capabilities and constraints"""
    
    # ROADM capabilities
    max_add_drop_cores: int = 4  # Maximum cores for add/drop
    wavelength_selective: bool = True
    colorless: bool = True
    directionless: bool = True
    contentionless: bool = True
    
    # Switching capabilities
    core_switching_enabled: bool = False  # Core switching not allowed per paper
    band_switching_enabled: bool = True   # Band switching allowed
    
    # Processing capabilities  
    max_grooming_capacity_gbps: int = 10000
    otn_switching: bool = True
    mpls_switching: bool = True
    
    # Physical constraints
    max_output_power_dbm: float = 20.0
    noise_figure_penalty_db: float = 0.5
    switching_penalty_db: float = 1.0

class MCFNode:
    """Multi-Core Fiber Network Node"""
    
    def __init__(self, node_id: int, node_name: str = None, 
                 capabilities: NodeCapabilities = None):
        """
        Initialize MCF Network Node
        
        Args:
            node_id: Unique node identifier
            node_name: Human-readable node name
            capabilities: Node capabilities object
        """
        self.node_id = node_id
        self.node_name = node_name or f"Node_{node_id}"
        self.capabilities = capabilities or NodeCapabilities()
        
        # Connected links
        self.incoming_links: Dict[int, 'MCFLink'] = {}  # {link_id: MCFLink}
        self.outgoing_links: Dict[int, 'MCFLink'] = {}  # {link_id: MCFLink}
        
        # Neighbor nodes
        self.neighbors: Set[int] = set()
        
        # Traffic demand tracking
        self.generated_requests: List[int] = []  # Request IDs originated here
        self.terminated_requests: List[int] = []  # Request IDs terminated here
        self.transit_requests: List[int] = []    # Request IDs passing through
        
        # Resource utilization
        self.add_drop_utilization: Dict[int, List[int]] = {core: [] for core in range(4)}
        self.switching_utilization: Dict[str, int] = {'add': 0, 'drop': 0, 'pass': 0}
        
        # QoT tracking
        self.node_penalties_db: Dict[str, float] = {
            'roadm_penalty': 0.5,
            'switching_penalty': 1.0,
            'connector_loss': 0.3
        }
        
    def add_link(self, link: 'MCFLink', direction: str):
        """
        Add a link to the node
        
        Args:
            link: MCFLink object
            direction: 'incoming' or 'outgoing'
        """
        if direction == 'incoming':
            self.incoming_links[link.link_id] = link
            self.neighbors.add(link.source)
        elif direction == 'outgoing':
            self.outgoing_links[link.link_id] = link
            self.neighbors.add(link.destination)
        else:
            raise ValueError("Direction must be 'incoming' or 'outgoing'")
    
    def get_link_to_neighbor(self, neighbor_id: int) -> 'MCFLink':
        """Get link to specific neighbor node"""
        for link in self.outgoing_links.values():
            if link.destination == neighbor_id:
                return link
        return None
    
    def get_link_from_neighbor(self, neighbor_id: int) -> 'MCFLink':
        """Get link from specific neighbor node"""
        for link in self.incoming_links.values():
            if link.source == neighbor_id:
                return link
        return None
    
    def can_add_request(self, cores_needed: List[int], 
                       bandwidth_gbps: int) -> bool:
        """
        Check if node can handle add operation for request
        
        Args:
            cores_needed: List of cores required
            bandwidth_gbps: Bandwidth requirement
            
        Returns:
            True if node can handle the add operation
        """
        # Check core availability
        if len(cores_needed) > self.capabilities.max_add_drop_cores:
            return False
            
        # Check grooming capacity
        current_add_capacity = sum(len(reqs) for reqs in self.add_drop_utilization.values())
        if current_add_capacity + bandwidth_gbps > self.capabilities.max_grooming_capacity_gbps:
            return False
            
        return True
    
    def can_drop_request(self, cores_used: List[int], 
                        bandwidth_gbps: int) -> bool:
        """
        Check if node can handle drop operation for request
        
        Args:
            cores_used: List of cores used by request
            bandwidth_gbps: Bandwidth to drop
            
        Returns:
            True if node can handle the drop operation
        """
        # Similar to add operation
        return self.can_add_request(cores_used, bandwidth_gbps)
    
    def add_request_tracking(self, request_id: int, operation: str, 
                           cores: List[int] = None):
        """
        Track request operations at this node
        
        Args:
            request_id: Request identifier
            operation: 'add', 'drop', or 'pass'
            cores: Cores involved in operation
        """
        if operation == 'add':
            self.generated_requests.append(request_id)
            self.switching_utilization['add'] += 1
            if cores:
                for core in cores:
                    self.add_drop_utilization[core].append(request_id)
                    
        elif operation == 'drop':
            self.terminated_requests.append(request_id)
            self.switching_utilization['drop'] += 1
            if cores:
                for core in cores:
                    if request_id in self.add_drop_utilization[core]:
                        self.add_drop_utilization[core].remove(request_id)
                        
        elif operation == 'pass':
            self.transit_requests.append(request_id)
            self.switching_utilization['pass'] += 1
    
    def remove_request_tracking(self, request_id: int):
        """Remove request from all tracking"""
        # Remove from generated/terminated lists
        if request_id in self.generated_requests:
            self.generated_requests.remove(request_id)
            self.switching_utilization['add'] -= 1
            
        if request_id in self.terminated_requests:
            self.terminated_requests.remove(request_id)
            self.switching_utilization['drop'] -= 1
            
        if request_id in self.transit_requests:
            self.transit_requests.remove(request_id)
            self.switching_utilization['pass'] -= 1
        
        # Remove from core utilization
        for core in range(4):
            if request_id in self.add_drop_utilization[core]:
                self.add_drop_utilization[core].remove(request_id)
    
    def get_node_penalty_db(self, operation: str) -> float:
        """
        Get QoT penalty for node operations
        
        Args:
            operation: 'add', 'drop', or 'pass'
            
        Returns:
            Penalty in dB
        """
        base_penalty = self.node_penalties_db['roadm_penalty']
        
        if operation in ['add', 'drop']:
            base_penalty += self.node_penalties_db['switching_penalty']
        
        # Add connector loss
        base_penalty += self.node_penalties_db['connector_loss']
        
        return base_penalty
    
    def get_utilization_statistics(self) -> Dict:
        """Get node utilization statistics"""
        total_requests = len(self.generated_requests) + len(self.terminated_requests) + len(self.transit_requests)
        
        core_utilization = {}
        for core in range(4):
            core_utilization[core] = {
                'active_requests': len(self.add_drop_utilization[core]),
                'utilization_ratio': len(self.add_drop_utilization[core]) / max(1, total_requests)
            }
        
        return {
            'node_id': self.node_id,
            'node_name': self.node_name,
            'total_requests_handled': total_requests,
            'generated_requests': len(self.generated_requests),
            'terminated_requests': len(self.terminated_requests),
            'transit_requests': len(self.transit_requests),
            'switching_operations': self.switching_utilization.copy(),
            'core_utilization': core_utilization,
            'neighbor_count': len(self.neighbors),
            'link_count': len(self.incoming_links) + len(self.outgoing_links)
        }
    
    def __str__(self):
        """String representation"""
        return f"MCFNode({self.node_id}: {self.node_name}, neighbors={list(self.neighbors)})"
    
    def __repr__(self):
        return self.__str__()