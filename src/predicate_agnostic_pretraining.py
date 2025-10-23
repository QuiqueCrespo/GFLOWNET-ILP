"""
Predicate-agnostic encoder pretraining.

Key Idea:
- Pretrain on GENERIC predicates (pred0, pred1, ..., pred_N)
- Encoder learns STRUCTURAL patterns (chain, fork, star, etc.)
- Predicate names are just indices - don't have semantic meaning
- During task: ANY vocabulary maps to indices → encoder recognizes structure

Example:
  Pretraining: "rule(X,Y) :- pred0(X,Z), pred0(Z,Y)" (chain structure)
  Task 1:      "grandparent(X,Y) :- parent(X,Z), parent(Y,Z)" (chain with "parent")
  Task 2:      "ancestor(X,Y) :- rel(X,Z), rel(Z,Y)" (chain with "rel")

  The encoder recognizes both as the SAME chain structure!

This is TRUE transfer learning - works with ANY vocabulary.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Dict

from .logic_structures import Variable, Atom, Rule, Theory, theory_to_string
from .graph_encoder import StateEncoder


class GenericRuleGenerator:
    """
    Generates random rules using GENERIC predicate names.

    Predicates are named: pred0, pred1, pred2, ..., pred_{N-1}
    This allows the encoder to learn structure-agnostic patterns.
    """

    def __init__(self, num_predicates: int = 10, max_arity: int = 2,
                 max_body_length: int = 4):
        """
        Args:
            num_predicates: Number of generic predicates (pred0, pred1, ...)
            max_arity: Maximum arity (1 or 2)
            max_body_length: Maximum body length
        """
        self.num_predicates = num_predicates
        self.max_arity = max_arity
        self.max_body_length = max_body_length

        # Generic predicate names
        self.predicate_names = [f'pred{i}' for i in range(num_predicates)]

        # Random arities for each predicate
        self.predicate_arities = {
            name: random.randint(1, max_arity)
            for name in self.predicate_names
        }

    def generate_random_rule(self, min_body_length: int = 1) -> Theory:
        """
        Generate a random valid rule with generic predicates.

        Returns:
            Theory with single rule using generic predicates
        """
        # Choose head predicate and arity
        head_arity = random.randint(1, self.max_arity)
        head_vars = [Variable(id=i) for i in range(head_arity)]
        head_pred = random.choice(self.predicate_names)
        head = Atom(predicate_name=head_pred, args=tuple(head_vars))

        # Generate body
        body_length = random.randint(min_body_length, self.max_body_length)
        body = []
        var_id = head_arity
        body_vars_set = set()

        for _ in range(body_length):
            # Choose predicate
            pred_name = random.choice(self.predicate_names)
            arity = self.predicate_arities[pred_name]

            # Create variables
            atom_vars = []
            for _ in range(arity):
                # Reuse existing or create new
                if body_vars_set and random.random() < 0.5:
                    var = random.choice(list(body_vars_set))
                else:
                    var = Variable(id=var_id)
                    var_id += 1
                    body_vars_set.add(var)
                atom_vars.append(var)

            atom = Atom(predicate_name=pred_name, args=tuple(atom_vars))
            body.append(atom)

        # Ensure no free variables
        head_vars_set = set(head_vars)
        if not head_vars_set.issubset(body_vars_set):
            # Connect head to body
            pred_name = random.choice(self.predicate_names)
            arity = self.predicate_arities[pred_name]

            missing_head_vars = list(head_vars_set - body_vars_set)
            available_body_vars = list(body_vars_set)

            atom_vars = []
            for i in range(arity):
                if missing_head_vars and random.random() < 0.7:
                    var = missing_head_vars.pop(0)
                elif available_body_vars:
                    var = random.choice(available_body_vars)
                else:
                    var = random.choice(head_vars)
                atom_vars.append(var)

            atom = Atom(predicate_name=pred_name, args=tuple(atom_vars))
            body.append(atom)

        return [Rule(head=head, body=body)]

    def generate_batch(self, batch_size: int) -> List[Theory]:
        """Generate batch of random rules."""
        return [self.generate_random_rule() for _ in range(batch_size)]


class StructuralAugmenter:
    """
    Creates structurally equivalent and different rule variations.

    Key: Only structural changes matter, not predicate names!
    """

    def __init__(self, num_predicates: int, predicate_arities: Dict[str, int]):
        self.num_predicates = num_predicates
        self.predicate_arities = predicate_arities
        self.predicate_names = list(predicate_arities.keys())

    # ==================== EQUIVALENT TRANSFORMATIONS ====================

    def rename_variables(self, theory: Theory) -> Theory:
        """Rename variables (structurally equivalent)."""
        if not theory:
            return theory

        rule = theory[0]
        from .logic_structures import get_all_variables

        old_vars = get_all_variables(theory)
        new_ids = random.sample(range(100, 100 + len(old_vars) * 2), len(old_vars))
        var_mapping = {old_var: Variable(id=new_id)
                      for old_var, new_id in zip(old_vars, new_ids)}

        def remap_args(args):
            return tuple(var_mapping.get(arg, arg) for arg in args)

        new_head = Atom(
            predicate_name=rule.head.predicate_name,
            args=remap_args(rule.head.args)
        )
        new_body = [
            Atom(predicate_name=atom.predicate_name, args=remap_args(atom.args))
            for atom in rule.body
        ]

        return [Rule(head=new_head, body=new_body)]

    def shuffle_body_atoms(self, theory: Theory) -> Theory:
        """Shuffle body atoms (structurally equivalent)."""
        if not theory:
            return theory

        rule = theory[0]
        shuffled_body = list(rule.body)
        random.shuffle(shuffled_body)

        return [Rule(head=rule.head, body=shuffled_body)]

    def duplicate_body_atom(self, theory: Theory) -> Theory:
        """Duplicate a body atom (structurally equivalent in logic)."""
        if not theory or not theory[0].body:
            return theory

        rule = theory[0]
        atom_to_dup = random.choice(rule.body)
        new_body = list(rule.body) + [atom_to_dup]

        return [Rule(head=rule.head, body=new_body)]

    # ==================== STRUCTURAL MODIFICATIONS ====================

    def replace_predicate_keep_structure(self, theory: Theory) -> Theory:
        """
        Replace a predicate with DIFFERENT one but keep structure.

        This changes the rule's identity but not its graph structure.
        Example: pred0(X,Y), pred0(Y,Z) → pred1(X,Y), pred1(Y,Z)
        Structure: chain remains chain, just different predicate
        """
        if not theory or not theory[0].body:
            return theory

        rule = theory[0]
        idx = random.randrange(len(rule.body))
        old_atom = rule.body[idx]

        # Choose different predicate with SAME arity
        same_arity_preds = [p for p in self.predicate_names
                           if self.predicate_arities[p] == len(old_atom.args)
                           and p != old_atom.predicate_name]

        if not same_arity_preds:
            return self.break_variable_connection(theory)

        new_pred = random.choice(same_arity_preds)
        new_atom = Atom(predicate_name=new_pred, args=old_atom.args)

        new_body = list(rule.body)
        new_body[idx] = new_atom

        return [Rule(head=rule.head, body=new_body)]

    def break_variable_connection(self, theory: Theory) -> Theory:
        """
        Break variable connections (CHANGES structure).

        Example:
          pred0(X,Y), pred1(Y,Z) → pred0(X,Y), pred1(W,Z)
          Structure: chain → disconnected (Y connection broken)
        """
        if not theory or not theory[0].body:
            return theory

        rule = theory[0]
        from .logic_structures import get_all_variables

        all_vars = get_all_variables(theory)
        if len(all_vars) < 2:
            return theory

        head_vars = set(rule.head.args)
        body_only_vars = [v for v in all_vars if v not in head_vars]

        if not body_only_vars:
            body_vars = set()
            for atom in rule.body:
                body_vars.update(atom.args)
            if not body_vars:
                return theory
            var_to_replace = random.choice(list(body_vars))
        else:
            var_to_replace = random.choice(body_only_vars)

        max_id = max(v.id for v in all_vars)
        new_var = Variable(id=max_id + 1)

        def replace_in_args(args):
            return tuple(new_var if arg == var_to_replace else arg for arg in args)

        new_body = [
            Atom(predicate_name=atom.predicate_name, args=replace_in_args(atom.args))
            for atom in rule.body
        ]

        return [Rule(head=rule.head, body=new_body)]

    def add_disconnected_atom(self, theory: Theory) -> Theory:
        """
        Add disconnected atom (CHANGES structure).

        Example:
          pred0(X,Y) → pred0(X,Y), pred1(Z,W)
          Structure: connected → has disconnected component
        """
        if not theory:
            return theory

        rule = theory[0]
        from .logic_structures import get_all_variables

        all_vars = get_all_variables(theory)
        max_id = max(v.id for v in all_vars) if all_vars else -1

        pred_name = random.choice(self.predicate_names)
        arity = self.predicate_arities[pred_name]

        # Create NEW variables (disconnected from existing)
        new_vars = [Variable(id=max_id + i + 1) for i in range(arity)]
        new_atom = Atom(predicate_name=pred_name, args=tuple(new_vars))

        new_body = list(rule.body) + [new_atom]
        return [Rule(head=rule.head, body=new_body)]

    def change_to_different_structure(self, theory: Theory) -> Theory:
        """
        Dramatically change structure.

        Examples:
          Chain → Star
          Fork → Chain
          etc.
        """
        transforms = [
            self.break_variable_connection,
            self.add_disconnected_atom,
            self.replace_predicate_keep_structure
        ]

        result = theory
        num_changes = random.randint(1, 2)
        for _ in range(num_changes):
            transform = random.choice(transforms)
            result = transform(result)

        return result

    def apply_equivalent_transform(self, theory: Theory) -> Theory:
        """Apply random structurally equivalent transformation."""
        transforms = [
            self.rename_variables,
            self.shuffle_body_atoms,
        ]

        if random.random() < 0.3 and theory and theory[0].body:
            transforms.append(self.duplicate_body_atom)

        result = theory
        num_transforms = random.randint(1, min(3, len(transforms)))
        chosen_transforms = random.sample(transforms, num_transforms)

        for transform in chosen_transforms:
            result = transform(result)

        return result

    def apply_structural_modification(self, theory: Theory) -> Theory:
        """Apply random structural modification."""
        return self.change_to_different_structure(theory)


class FlexibleGraphConstructor:
    """
    Graph constructor that works with ANY vocabulary.

    Key idea: Map predicates to indices dynamically.
    - Pretraining: uses indices 0, 1, 2, ..., N-1 for pred0, pred1, ...
    - Task: uses indices 0, 1, 2, ..., M-1 for task predicates

    The encoder sees the same structure regardless of vocabulary!
    """

    def __init__(self, max_predicates: int = 20):
        """
        Args:
            max_predicates: Maximum number of predicates (fixed capacity)
        """
        self.max_predicates = max_predicates

    def theory_to_graph(self, theory: Theory, predicate_vocab: List[str] = None) -> 'Data':
        """
        Convert theory to graph with dynamic predicate mapping.

        Args:
            theory: Theory to encode
            predicate_vocab: List of predicates in this vocabulary
                           If None, extracts from theory

        Returns:
            PyTorch Geometric Data object
        """
        from torch_geometric.data import Data

        if not theory:
            # Empty theory: create dummy predicate node with correct feature dimension
            # Predicate features: max_predicates + 7 (to match EnhancedGraphConstructor)
            var_dim = 6
            pred_dim = self.max_predicates + 7
            max_dim = max(var_dim, pred_dim)
            return Data(
                x=torch.zeros((1, max_dim)),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 3), dtype=torch.float),  # Add edge_attr
                num_nodes=1,
                is_variable=torch.tensor([False]),
                is_head=torch.tensor([False]),
                is_body=torch.tensor([False])
            )

        # Extract vocabulary from theory if not provided
        if predicate_vocab is None:
            predicate_vocab = list(set(
                [rule.head.predicate_name for rule in theory] +
                [atom.predicate_name for rule in theory for atom in rule.body]
            ))

        # Create predicate to index mapping
        pred_to_idx = {pred: idx for idx, pred in enumerate(predicate_vocab)}

        node_features = []
        node_types = []
        edges = []
        edge_features = []  # Edge features: [arg_position, is_head_edge, is_body_edge]
        node_id = 0

        # Get canonical variable ordering
        from .logic_structures import get_all_variables
        variables = get_all_variables(theory)

        # Analyze variable usage patterns (needed for structural features)
        head_vars = set()
        body_vars = set()
        var_counts = {}

        for rule in theory:
            for var in rule.head.args:
                head_vars.add(var)
                var_counts[var] = var_counts.get(var, 0) + 1
            for atom in rule.body:
                for var in atom.args:
                    body_vars.add(var)
                    var_counts[var] = var_counts.get(var, 0) + 1

        var_to_node = {}
        for var in variables:
            var_to_node[var] = node_id
            node_types.append('var')

            # Variable features (6 features total to match EnhancedGraphConstructor)
            var_feature = [
                1.0 if var in head_vars else 0.0,              # appears_in_head
                1.0 if var in body_vars else 0.0,              # appears_in_body
                1.0 if var_counts.get(var, 0) > 1 else 0.0,   # appears_multiple
                1.0 if (var in body_vars and var not in head_vars) else 0.0,  # is_chain_var
                float(var_counts.get(var, 0)),                 # total_occurrences
                1.0                                             # is_variable flag
            ]
            node_features.append(var_feature)
            node_id += 1

        # Process rules
        for rule_idx, rule in enumerate(theory):
            # Head predicate
            head_node = node_id
            node_types.append('pred')

            # Predicate features (max_predicates + 7 to match EnhancedGraphConstructor)
            # [one_hot, is_head, is_body, has_self_loop, num_unique_vars, total_vars, is_variable]
            pred_one_hot = [0.0] * self.max_predicates
            if rule.head.predicate_name in pred_to_idx:
                idx = pred_to_idx[rule.head.predicate_name]
                if idx < self.max_predicates:
                    pred_one_hot[idx] = 1.0

            head_has_self_loop = len(set(rule.head.args)) < len(rule.head.args)
            head_features = pred_one_hot + [
                1.0,                                      # is_head
                0.0,                                      # is_body
                1.0 if head_has_self_loop else 0.0,     # has_self_loop
                float(len(set(rule.head.args))),         # num_unique_vars
                float(len(rule.head.args)),              # total_vars
                0.0                                       # is_variable flag
            ]
            node_features.append(head_features)
            node_id += 1

            # Connect to arguments with edge features
            for arg_pos, arg in enumerate(rule.head.args):
                if isinstance(arg, Variable) and arg in var_to_node:
                    var_node = var_to_node[arg]
                    edges.append([var_node, head_node])
                    edges.append([head_node, var_node])
                    # Edge features: [argument_position, is_head_edge, is_body_edge]
                    edge_feat = [float(arg_pos), 1.0, 0.0]
                    edge_features.extend([edge_feat, edge_feat])  # Both directions

            # Body atoms
            for atom_idx, atom in enumerate(rule.body):
                atom_node = node_id
                node_types.append('pred')

                # Predicate one-hot
                pred_one_hot = [0.0] * self.max_predicates
                if atom.predicate_name in pred_to_idx:
                    idx = pred_to_idx[atom.predicate_name]
                    if idx < self.max_predicates:
                        pred_one_hot[idx] = 1.0

                atom_has_self_loop = len(set(atom.args)) < len(atom.args)
                body_features = pred_one_hot + [
                    0.0,                                      # is_head
                    1.0,                                      # is_body
                    1.0 if atom_has_self_loop else 0.0,     # has_self_loop
                    float(len(set(atom.args))),              # num_unique_vars
                    float(len(atom.args)),                   # total_vars
                    0.0                                       # is_variable flag
                ]
                node_features.append(body_features)
                node_id += 1

                for arg_pos, arg in enumerate(atom.args):
                    if isinstance(arg, Variable) and arg in var_to_node:
                        var_node = var_to_node[arg]
                        edges.append([var_node, atom_node])
                        edges.append([atom_node, var_node])
                        # Edge features: [argument_position, is_head_edge, is_body_edge]
                        edge_feat = [float(arg_pos), 0.0, 1.0]
                        edge_features.extend([edge_feat, edge_feat])  # Both directions

        # Create masks for hierarchical pooling BEFORE padding
        # (matches EnhancedGraphConstructor's behavior)
        num_nodes = len(node_features)
        num_vars = len(var_to_node)

        # is_variable: check last element of original features (before padding)
        is_variable = torch.tensor([f[-1] == 1.0 for f in node_features])

        # is_head: True for head predicate node (first predicate after variables)
        is_head = torch.zeros(num_nodes, dtype=torch.bool)
        if num_nodes > num_vars:
            is_head[num_vars] = True

        # is_body: True for body predicate nodes (all predicates except head)
        is_body = torch.zeros(num_nodes, dtype=torch.bool)
        if num_nodes > num_vars + 1:
            is_body[num_vars + 1:] = True

        # Pad features to same length (max of var and pred dimensions)
        # This matches EnhancedGraphConstructor's behavior
        var_dim = 6
        pred_dim = self.max_predicates + 7
        max_dim = max(var_dim, pred_dim)

        # Pad all features to max_dim
        padded_features = []
        for feat in node_features:
            if len(feat) < max_dim:
                feat = feat + [0.0] * (max_dim - len(feat))
            padded_features.append(feat)

        x = torch.tensor(padded_features, dtype=torch.float)

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)

        # Create edge_attr tensor
        if edge_features:
            edge_attr = torch.tensor(edge_features, dtype=torch.float)
        else:
            edge_attr = torch.empty((0, 3), dtype=torch.float)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,  # Add edge attributes
            num_nodes=num_nodes,
            is_variable=is_variable,
            is_head=is_head,
            is_body=is_body
        )


class PredicateAgnosticPretrainer:
    """
    Pretrain encoder on generic predicates with contrastive learning.

    The encoder learns to recognize structural patterns:
    - Chains: A(x,y), A(y,z)
    - Forks: A(x,z), B(y,z)
    - Stars: A(x,y), A(x,z), A(x,w)
    - etc.

    These patterns transfer to ANY vocabulary!
    """

    def __init__(self,
                 state_encoder: StateEncoder,
                 num_generic_predicates: int = 10,
                 max_predicates: int = None,
                 max_arity: int = 2,
                 learning_rate: float = 1e-3,
                 temperature: float = 0.5,
                 num_negatives: int = 4):
        """
        Args:
            state_encoder: GNN encoder to pretrain
            num_generic_predicates: Number of generic predicates (pred0, ...)
            max_predicates: Maximum capacity for predicates in graph encoding
                          (should match StateEncoder's expected input dimension - 1)
                          If None, uses num_generic_predicates
            max_arity: Maximum predicate arity
            learning_rate: Learning rate
            temperature: Temperature for contrastive loss
            num_negatives: Number of negative samples
        """
        self.encoder = state_encoder
        self.num_generic_predicates = num_generic_predicates

        # Infer max_predicates from state_encoder if not provided
        if max_predicates is None:
            # Try to get from encoder's expected input dimension
            if hasattr(state_encoder, 'input_dim'):
                max_predicates = state_encoder.input_dim - 1
            elif hasattr(state_encoder, 'predicate_vocab_size'):
                max_predicates = state_encoder.predicate_vocab_size
            else:
                max_predicates = num_generic_predicates

        self.max_predicates = max_predicates

        self.rule_generator = GenericRuleGenerator(
            num_predicates=num_generic_predicates,
            max_arity=max_arity
        )

        self.augmenter = StructuralAugmenter(
            num_predicates=num_generic_predicates,
            predicate_arities=self.rule_generator.predicate_arities
        )

        self.graph_constructor = FlexibleGraphConstructor(
            max_predicates=max_predicates
        )

        self.generic_predicate_vocab = self.rule_generator.predicate_names

        # Contrastive loss
        from .encoder_pretraining import ContrastiveLoss
        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.num_negatives = num_negatives

        # Optimizer
        self.optimizer = torch.optim.Adam(state_encoder.parameters(), lr=learning_rate)

    def create_training_batch(self, batch_size: int):
        """Create batch of (anchor, positive, negatives)."""
        anchors = self.rule_generator.generate_batch(batch_size)

        # Positives: equivalent transformations
        positives = [self.augmenter.apply_equivalent_transform(anchor)
                     for anchor in anchors]

        # Negatives: structural modifications
        negatives = []
        for anchor in anchors:
            neg_samples = []
            for _ in range(self.num_negatives):
                if random.random() < 0.6:
                    neg = self.augmenter.apply_structural_modification(anchor)
                else:
                    neg = self.rule_generator.generate_random_rule()
                neg_samples.append(neg)
            negatives.append(neg_samples)

        return anchors, positives, negatives

    def encode_theory(self, theory: Theory) -> torch.Tensor:
        """Encode theory using generic predicate vocabulary."""
        graph_data = self.graph_constructor.theory_to_graph(
            theory,
            predicate_vocab=self.generic_predicate_vocab
        )
        graph_embedding, _ = self.encoder(graph_data)
        return graph_embedding.squeeze(0)

    def pretrain_step(self, batch_size: int) -> Dict[str, float]:
        """One pretraining step."""
        anchors, positives, negatives = self.create_training_batch(batch_size)

        # Encode
        anchor_embeds = torch.stack([self.encode_theory(a) for a in anchors])
        positive_embeds = torch.stack([self.encode_theory(p) for p in positives])

        negative_embeds = []
        for neg_list in negatives:
            neg_embeds_i = torch.stack([self.encode_theory(neg) for neg in neg_list])
            negative_embeds.append(neg_embeds_i)
        negative_embeds = torch.stack(negative_embeds)

        # Loss
        loss = self.contrastive_loss(anchor_embeds, positive_embeds, negative_embeds)

        # Backward
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Metrics
        with torch.no_grad():
            anchor_norm = F.normalize(anchor_embeds, dim=1)
            positive_norm = F.normalize(positive_embeds, dim=1)
            negative_norm = F.normalize(negative_embeds, dim=2)

            pos_sim = torch.sum(anchor_norm * positive_norm, dim=1)
            neg_sim = torch.bmm(negative_norm, anchor_norm.unsqueeze(2)).squeeze(2)

            max_neg_sim = torch.max(neg_sim, dim=1)[0]
            accuracy = (pos_sim > max_neg_sim).float().mean().item()

        return {
            'loss': loss.item(),
            'accuracy': accuracy,
            'avg_pos_sim': pos_sim.mean().item(),
            'avg_neg_sim': neg_sim.mean().item()
        }

    def pretrain(self, num_steps: int, batch_size: int = 32,
                 verbose: bool = True, log_interval: int = 100) -> List[Dict[str, float]]:
        """Pretrain encoder."""
        history = []

        for step in range(num_steps):
            metrics = self.pretrain_step(batch_size)
            history.append(metrics)

            if verbose and (step % log_interval == 0 or step == num_steps - 1):
                print(f"Step {step:4d}/{num_steps} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Acc: {metrics['accuracy']:.3f} | "
                      f"Pos Sim: {metrics['avg_pos_sim']:.3f} | "
                      f"Neg Sim: {metrics['avg_neg_sim']:.3f}")

        return history

    def save_pretrained_encoder(self, path: str):
        """Save pretrained weights."""
        torch.save(self.encoder.state_dict(), path)
        print(f"Saved pretrained encoder to {path}")

    def load_pretrained_encoder(self, path: str):
        """Load pretrained weights."""
        self.encoder.load_state_dict(torch.load(path))
        print(f"Loaded pretrained encoder from {path}")
