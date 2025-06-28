import logging
import re
from typing import Dict, List, Any, Optional, Tuple, Set, Union
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)

class KnowledgeGraph:
    def __init__(self, llm: Optional[Any] = None):
        self.llm = llm
        self.graph = nx.DiGraph()
        self.entity_metadata = {}
        self.relationship_metadata = {}

    def add_entity(self, entity_id: str, entity_type: str, metadata: Dict[str, Any] = None) -> None:
        if not self.graph.has_node(entity_id):
            self.graph.add_node(entity_id)
            self.entity_metadata[entity_id] = {
                'type': entity_type,
                'metadata': metadata or {}
            }
            logger.debug(f"Added entity: {entity_id} of type {entity_type}")
        else:
            self.entity_metadata[entity_id]['type'] = entity_type
            if metadata:
                self.entity_metadata[entity_id]['metadata'].update(metadata)
            logger.debug(f"Updated entity: {entity_id}")

    def add_relationship(self, source_id: str, target_id: str, relationship_type: str,
                           weight: float = 1.0, metadata: Dict[str, Any] = None) -> None:
        if not self.graph.has_node(source_id):
            self.add_entity(source_id, 'unknown')
        if not self.graph.has_node(target_id):
            self.add_entity(target_id, 'unknown')

        self.graph.add_edge(source_id, target_id,
                            relationship_type=relationship_type,
                            weight=weight)

        rel_key = (source_id, target_id)
        self.relationship_metadata[rel_key] = {
            'type': relationship_type,
            'weight': weight,
            'metadata': metadata or {}
        }

        logger.debug(f"Added relationship: {source_id} --[{relationship_type}]--> {target_id} (weight: {weight})")

    def extract_entities_and_relationships(self, text: str) -> Dict[str, Any]:
        result = {
            'entities': [],
            'relationships': [],
            'error': None
        }

        try:
            if not self.llm:
                raise ValueError("Language model not initialized. Cannot extract entities and relationships.")

            messages = [
                SystemMessage(content="You are an expert in knowledge extraction. "
                                      "Extract entities and relationships from the provided text. "
                                      "For each entity, identify its type and any relevant attributes. "
                                      "For each relationship, identify the source entity, target entity, relationship type, and confidence. "
                                      "Format your response as a structured JSON object with 'entities' and 'relationships' arrays."),
                HumanMessage(content=f"Text: {text}\n\nExtract entities and relationships in JSON format.")
            ]

            response = self.llm.generate([messages])
            extraction_text = response.generations[0][0].text.strip()

            import json
            try:
                json_match = re.search(r'\{[\s\S]*\}', extraction_text)
                if json_match:
                    extraction_json = json.loads(json_match.group(0))

                    if 'entities' in extraction_json:
                        result['entities'] = extraction_json['entities']
                        for entity in extraction_json['entities']:
                            if 'id' in entity and 'type' in entity:
                                self.add_entity(
                                    entity_id=entity['id'],
                                    entity_type=entity['type'],
                                    metadata={k: v for k, v in entity.items() if k not in ['id', 'type']}
                                )

                    if 'relationships' in extraction_json:
                        result['relationships'] = extraction_json['relationships']
                        for rel in extraction_json['relationships']:
                            if all(k in rel for k in ['source', 'target', 'type']):
                                weight = rel.get('confidence', 1.0)
                                self.add_relationship(
                                    source_id=rel['source'],
                                    target_id=rel['target'],
                                    relationship_type=rel['type'],
                                    weight=weight,
                                    metadata={k: v for k, v in rel.items() if k not in ['source', 'target', 'type', 'confidence']}
                                )
                else:
                    result['error'] = "Could not find JSON in the extraction response."
            except json.JSONDecodeError as e:
                result['error'] = f"Error parsing extraction JSON: {str(e)}"
                result['raw_response'] = extraction_text

        except Exception as e:
            logger.error(f"Error extracting entities and relationships: {str(e)}")
            result['error'] = str(e)

        return result

    def build_from_text(self, text: str) -> Dict[str, Any]:
        return self.extract_entities_and_relationships(text)

    def build_from_texts(self, texts: List[str]) -> Dict[str, Any]:
        results = {
            'entities': [],
            'relationships': [],
            'errors': []
        }

        for i, text in enumerate(texts):
            result = self.build_from_text(text)
            results['entities'].extend(result.get('entities', []))
            results['relationships'].extend(result.get('relationships', []))
            if result.get('error'):
                results['errors'].append({
                    'text_index': i,
                    'error': result['error']
                })

        return results

    def get_entity(self, entity_id: str) -> Dict[str, Any]:
        if not self.graph.has_node(entity_id):
            return None

        entity_info = self.entity_metadata.get(entity_id, {})

        incoming_relationships = []
        outgoing_relationships = []

        for source, target in self.graph.in_edges(entity_id):
            rel_key = (source, target)
            rel_info = self.relationship_metadata.get(rel_key, {})
            incoming_relationships.append({
                'source': source,
                'type': rel_info.get('type', 'unknown'),
                'weight': rel_info.get('weight', 1.0),
                'metadata': rel_info.get('metadata', {})
            })

        for source, target in self.graph.out_edges(entity_id):
            rel_key = (source, target)
            rel_info = self.relationship_metadata.get(rel_key, {})
            outgoing_relationships.append({
                'target': target,
                'type': rel_info.get('type', 'unknown'),
                'weight': rel_info.get('weight', 1.0),
                'metadata': rel_info.get('metadata', {})
            })

        return {
            'id': entity_id,
            'type': entity_info.get('type', 'unknown'),
            'metadata': entity_info.get('metadata', {}),
            'incoming_relationships': incoming_relationships,
            'outgoing_relationships': outgoing_relationships
        }

    def get_relationship(self, source_id: str, target_id: str) -> Dict[str, Any]:
        if not self.graph.has_edge(source_id, target_id):
            return None

        rel_key = (source_id, target_id)
        rel_info = self.relationship_metadata.get(rel_key, {})

        return {
            'source': source_id,
            'target': target_id,
            'type': rel_info.get('type', 'unknown'),
            'weight': rel_info.get('weight', 1.0),
            'metadata': rel_info.get('metadata', {})
        }

    def find_paths(self, source_id: str, target_id: str, max_length: int = 5) -> List[List[Tuple[str, str, str]]]:
        if not self.graph.has_node(source_id) or not self.graph.has_node(target_id):
            return []

        try:
            paths = list(nx.all_simple_paths(self.graph, source_id, target_id, cutoff=max_length))
        except nx.NetworkXNoPath:
            return []

        result_paths = []
        for path in paths:
            path_relationships = []
            for i in range(len(path) - 1):
                source = path[i]
                target = path[i + 1]
                rel_key = (source, target)
                rel_type = self.relationship_metadata.get(rel_key, {}).get('type', 'unknown')
                path_relationships.append((source, target, rel_type))
            result_paths.append(path_relationships)

        return result_paths

    def find_causal_paths(self, source_id: str, target_id: str, max_length: int = 5) -> List[Dict[str, Any]]:
        all_paths = self.find_paths(source_id, target_id, max_length)
        causal_paths = []

        for path in all_paths:
            is_causal = all(rel[2] in ['causes', 'influences', 'affects', 'leads_to', 'results_in', 'contributes_to']
                            for rel in path)

            if is_causal:
                path_strength = 1.0
                path_details = []

                for source, target, rel_type in path:
                    rel_key = (source, target)
                    weight = self.relationship_metadata.get(rel_key, {}).get('weight', 1.0)
                    path_strength *= weight

                    source_name = self.entity_metadata.get(source, {}).get('metadata', {}).get('name', source)
                    target_name = self.entity_metadata.get(target, {}).get('metadata', {}).get('name', target)

                    path_details.append({
                        'source': source,
                        'source_name': source_name,
                        'target': target,
                        'target_name': target_name,
                        'relationship': rel_type,
                        'weight': weight
                    })

                causal_paths.append({
                    'path': path_details,
                    'strength': path_strength,
                    'length': len(path)
                })

        causal_paths.sort(key=lambda x: x['strength'], reverse=True)

        return causal_paths

    def find_common_causes(self, entity_ids: List[str]) -> Dict[str, float]:
        if not entity_ids or len(entity_ids) < 2:
            return {}

        entity_causes = {}
        for entity_id in entity_ids:
            causes = {}
            for source, _ in self.graph.in_edges(entity_id):
                rel_key = (source, entity_id)
                rel_info = self.relationship_metadata.get(rel_key, {})
                if rel_info.get('type') in ['causes', 'influences', 'affects', 'leads_to', 'results_in', 'contributes_to']:
                    causes[source] = rel_info.get('weight', 1.0)
            entity_causes[entity_id] = causes

        common_causes = {}
        all_causes = set()
        for causes in entity_causes.values():
            all_causes.update(causes.keys())

        for cause in all_causes:
            if all(cause in causes for causes in entity_causes.values()):
                avg_strength = sum(causes[cause] for causes in entity_causes.values() if cause in causes) / len(entity_ids)
                common_causes[cause] = avg_strength

        common_causes = dict(sorted(common_causes.items(), key=lambda x: x[1], reverse=True))

        return common_causes

    def find_common_effects(self, entity_ids: List[str]) -> Dict[str, float]:
        if not entity_ids or len(entity_ids) < 2:
            return {}

        entity_effects = {}
        for entity_id in entity_ids:
            effects = {}
            for _, target in self.graph.out_edges(entity_id):
                rel_key = (entity_id, target)
                rel_info = self.relationship_metadata.get(rel_key, {})
                if rel_info.get('type') in ['causes', 'influences', 'affects', 'leads_to', 'results_in', 'contributes_to']:
                    effects[target] = rel_info.get('weight', 1.0)
            entity_effects[entity_id] = effects

        common_effects = {}
        all_effects = set()
        for effects in entity_effects.values():
            all_effects.update(effects.keys())

        for effect in all_effects:
            if all(effect in effects for effects in entity_effects.values()):
                avg_strength = sum(effects[effect] for effects in entity_effects.values() if effect in effects) / len(entity_ids)
                common_effects[effect] = avg_strength

        common_effects = dict(sorted(common_effects.items(), key=lambda x: x[1], reverse=True))

        return common_effects

    def predict_effects(self, entity_id: str, max_depth: int = 3) -> Dict[str, Dict[str, Any]]:
        if not self.graph.has_node(entity_id):
            return {}

        effects = {}
        visited = set()

        def dfs(node, depth, path, strength):
            if depth > max_depth or node in visited:
                return

            visited.add(node)

            for _, target in self.graph.out_edges(node):
                rel_key = (node, target)
                rel_info = self.relationship_metadata.get(rel_key, {})
                rel_type = rel_info.get('type', 'unknown')
                weight = rel_info.get('weight', 1.0)

                if rel_type in ['causes', 'influences', 'affects', 'leads_to', 'results_in', 'contributes_to']:
                    new_strength = strength * weight
                    new_path = path + [(node, target, rel_type, weight)]

                    if target in effects:
                        if new_strength > effects[target]['strength']:
                            effects[target] = {
                                'strength': new_strength,
                                'path': new_path,
                                'depth': depth
                            }
                    else:
                        effects[target] = {
                            'strength': new_strength,
                            'path': new_path,
                            'depth': depth
                        }

                    dfs(target, depth + 1, new_path, new_strength)

            visited.remove(node)

        dfs(entity_id, 1, [], 1.0)

        formatted_effects = {}
        for effect_id, details in effects.items():
            effect_name = self.entity_metadata.get(effect_id, {}).get('metadata', {}).get('name', effect_id)
            effect_type = self.entity_metadata.get(effect_id, {}).get('type', 'unknown')

            formatted_path = []
            for source, target, rel_type, weight in details['path']:
                source_name = self.entity_metadata.get(source, {}).get('metadata', {}).get('name', source)
                target_name = self.entity_metadata.get(target, {}).get('metadata', {}).get('name', target)

                formatted_path.append({
                    'source': source,
                    'source_name': source_name,
                    'target': target,
                    'target_name': target_name,
                    'relationship': rel_type,
                    'weight': weight
                })

            formatted_effects[effect_id] = {
                'id': effect_id,
                'name': effect_name,
                'type': effect_type,
                'strength': details['strength'],
                'depth': details['depth'],
                'path': formatted_path
            }

        formatted_effects = dict(sorted(formatted_effects.items(), key=lambda x: x[1]['strength'], reverse=True))

        return formatted_effects

    def predict_causes(self, entity_id: str, max_depth: int = 3) -> Dict[str, Dict[str, Any]]:
        if not self.graph.has_node(entity_id):
            return {}

        causes = {}
        visited = set()

        def dfs(node, depth, path, strength):
            if depth > max_depth or node in visited:
                return

            visited.add(node)

            for source, _ in self.graph.in_edges(node):
                rel_key = (source, node)
                rel_info = self.relationship_metadata.get(rel_key, {})
                rel_type = rel_info.get('type', 'unknown')
                weight = rel_info.get('weight', 1.0)

                if rel_type in ['causes', 'influences', 'affects', 'leads_to', 'results_in', 'contributes_to']:
                    new_strength = strength * weight
                    new_path = [(source, node, rel_type, weight)] + path

                    if source in causes:
                        if new_strength > causes[source]['strength']:
                            causes[source] = {
                                'strength': new_strength,
                                'path': new_path,
                                'depth': depth
                            }
                    else:
                        causes[source] = {
                            'strength': new_strength,
                            'path': new_path,
                            'depth': depth
                        }

                    dfs(source, depth + 1, new_path, new_strength)

            visited.remove(node)

        dfs(entity_id, 1, [], 1.0)

        formatted_causes = {}
        for cause_id, details in causes.items():
            cause_name = self.entity_metadata.get(cause_id, {}).get('metadata', {}).get('name', cause_id)
            cause_type = self.entity_metadata.get(cause_id, {}).get('type', 'unknown')

            formatted_path = []
            for source, target, rel_type, weight in details['path']:
                source_name = self.entity_metadata.get(source, {}).get('metadata', {}).get('name', source)
                target_name = self.entity_metadata.get(target, {}).get('metadata', {}).get('name', target)

                formatted_path.append({
                    'source': source,
                    'source_name': source_name,
                    'target': target,
                    'target_name': target_name,
                    'relationship': rel_type,
                    'weight': weight
                })

            formatted_causes[cause_id] = {
                'id': cause_id,
                'name': cause_name,
                'type': cause_type,
                'strength': details['strength'],
                'depth': details['depth'],
                'path': formatted_path
            }

        formatted_causes = dict(sorted(formatted_causes.items(), key=lambda x: x[1]['strength'], reverse=True))

        return formatted_causes

    def suggest_interventions(self, target_id: str, desired_state: str = 'increase') -> List[Dict[str, Any]]:
        if not self.graph.has_node(target_id):
            return []

        causes = self.predict_causes(target_id, max_depth=3)

        interventions = []
        for cause_id, cause_info in causes.items():
            path = cause_info['path']
            cumulative_effect = 1.0

            for step in path:
                rel_type = step['relationship']
                weight = step['weight']

                is_positive = rel_type not in ['inhibits', 'decreases', 'prevents', 'reduces']

                if not is_positive:
                    cumulative_effect *= -1

            intervention_type = 'increase' if (
                (desired_state == 'increase' and cumulative_effect > 0) or
                (desired_state == 'decrease' and cumulative_effect < 0)
            ) else 'decrease'

            intervention_score = cause_info['strength'] * abs(cumulative_effect)

            interventions.append({
                'entity_id': cause_id,
                'entity_name': cause_info['name'],
                'entity_type': cause_info['type'],
                'intervention': intervention_type,
                'score': intervention_score,
                'path': cause_info['path'],
                'depth': cause_info['depth']
            })

        interventions.sort(key=lambda x: x['score'], reverse=True)

        return interventions

    def visualize(self, entity_ids: Optional[List[str]] = None,
                  max_nodes: int = 20,
                  include_attributes: bool = False,
                  highlight_path: Optional[List[Tuple[str, str]]] = None) -> plt.Figure:
        if entity_ids:
            all_nodes = set(entity_ids)
            for entity_id in entity_ids:
                if self.graph.has_node(entity_id):
                    all_nodes.update(self.graph.predecessors(entity_id))
                    all_nodes.update(self.graph.successors(entity_id))

            if len(all_nodes) > max_nodes:
                extra_nodes = list(all_nodes - set(entity_ids))
                extra_nodes.sort(key=lambda x: self.graph.degree(x), reverse=True)
                all_nodes = set(entity_ids) | set(extra_nodes[:max_nodes - len(entity_ids)])

            subgraph = self.graph.subgraph(all_nodes)
        else:
            if len(self.graph) > max_nodes:
                sorted_nodes = sorted(self.graph.nodes(), key=lambda x: self.graph.degree(x), reverse=True)
                subgraph = self.graph.subgraph(sorted_nodes[:max_nodes])
            else:
                subgraph = self.graph

        plt.figure(figsize=(12, 10))

        node_labels = {}
        for node in subgraph.nodes():
            entity_info = self.entity_metadata.get(node, {})
            entity_name = entity_info.get('metadata', {}).get('name', node)
            entity_type = entity_info.get('type', 'unknown')

            if include_attributes:
                attrs = []
                for k, v in entity_info.get('metadata', {}).items():
                    if k != 'name' and len(str(v)) < 20:
                        attrs.append(f"{k}: {v}")

                if attrs:
                    node_labels[node] = f"{entity_name}\n({entity_type})\n{', '.join(attrs)}"
                else:
                    node_labels[node] = f"{entity_name}\n({entity_type})"
            else:
                node_labels[node] = f"{entity_name}\n({entity_type})"

        edge_labels = {}
        for source, target in subgraph.edges():
            rel_key = (source, target)
            rel_info = self.relationship_metadata.get(rel_key, {})
            rel_type = rel_info.get('type', 'unknown')
            weight = rel_info.get('weight', 1.0)

            edge_labels[(source, target)] = f"{rel_type}\n({weight:.2f})"

        node_colors = []
        entity_types = set(info.get('type', 'unknown') for info in self.entity_metadata.values())
        color_map = plt.cm.get_cmap('tab10', len(entity_types))
        type_to_color = {t: color_map(i) for i, t in enumerate(entity_types)}

        for node in subgraph.nodes():
            entity_type = self.entity_metadata.get(node, {}).get('type', 'unknown')