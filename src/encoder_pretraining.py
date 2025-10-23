"""
Pretraining module for the GNN encoder using contrastive learning.

The idea is to pretrain the encoder to understand logical rule structure by:
1. Generating random rules
2. Creating equivalent rules (variable renaming, atom reordering, etc.)
3. Creating semantically different rules (atom substitution, variable changes, etc.)
4. Training with contrastive loss to bring equivalent rules closer and push different rules apart

Theoretical Justification:
- Contrastive learning learns invariances to semantically-preserving transformations
- Pretraining on diverse synthetic rules improves generalization to real ILP tasks
- Similar to BERT pretraining for NLP, but for logical structures
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
from typing import List, Tuple, Dict, Set
from collections import defaultdict

from .logic_structures import Variable, Atom, Rule, Theory, theory_to_string
from .graph_encoder import GraphConstructor, StateEncoder


class RandomRuleGenerator:
    """Generates random logical rules for pretraining."""

    def __init__(self, predicate_vocab: List[str], predicate_arities: Dict[str, int],
                 max_body_length: int = 4, max_head_arity: int = 2):
        """
        Args:
            predicate_vocab: List of available predicates
            predicate_arities: Mapping from predicate to arity (1 or 2)
            max_body_length: Maximum number of atoms in rule body
            max_head_arity: Maximum arity of head predicate
        """
        self.predicate_vocab = predicate_vocab
        self.predicate_arities = predicate_arities
        self.max_body_length = max_body_length
        self.max_head_arity = max_head_arity

    def generate_random_rule(self, min_body_length: int = 1) -> Theory:
        """
        Generate a random valid rule.

        Ensures:
        - No free variables (all head variables appear in body)
        - At least min_body_length atoms in body
        - Uses valid predicates from vocabulary

        Returns:
            Theory containing a single random rule
        """
        # Choose head predicate and arity
        head_arity = random.randint(1, self.max_head_arity)
        head_vars = [Variable(id=i) for i in range(head_arity)]
        head_pred = random.choice(self.predicate_vocab)
        head = Atom(predicate_name=head_pred, args=tuple(head_vars))

        # Generate body atoms
        body_length = random.randint(min_body_length, self.max_body_length)
        body = []
        var_id = head_arity  # Start variable IDs after head variables

        # Track which variables we've used in the body
        body_vars_set = set()

        for _ in range(body_length):
            # Choose predicate
            pred_name = random.choice(self.predicate_vocab)
            arity = self.predicate_arities[pred_name]

            # Create variables for this atom
            atom_vars = []
            for _ in range(arity):
                # Randomly decide: use existing variable or create new one
                if body_vars_set and random.random() < 0.5:
                    # Reuse existing variable (promotes connectivity)
                    var = random.choice(list(body_vars_set))
                else:
                    # Create new variable
                    var = Variable(id=var_id)
                    var_id += 1
                    body_vars_set.add(var)
                atom_vars.append(var)

            atom = Atom(predicate_name=pred_name, args=tuple(atom_vars))
            body.append(atom)

        # Ensure all head variables appear in body (no free variables)
        head_vars_set = set(head_vars)
        if not head_vars_set.issubset(body_vars_set):
            # Add one more atom connecting head variables to body
            pred_name = random.choice(self.predicate_vocab)
            arity = self.predicate_arities[pred_name]

            # Use head variables and body variables
            atom_vars = []
            missing_head_vars = list(head_vars_set - body_vars_set)
            available_body_vars = list(body_vars_set)

            for i in range(arity):
                if missing_head_vars and random.random() < 0.7:
                    # Use missing head variable
                    var = missing_head_vars.pop(0)
                elif available_body_vars:
                    var = random.choice(available_body_vars)
                else:
                    var = random.choice(head_vars)
                atom_vars.append(var)

            atom = Atom(predicate_name=pred_name, args=tuple(atom_vars))
            body.append(atom)

        rule = Rule(head=head, body=body)
        return [rule]

    def generate_batch(self, batch_size: int) -> List[Theory]:
        """Generate a batch of random rules."""
        return [self.generate_random_rule() for _ in range(batch_size)]


class RuleAugmenter:
    """Creates semantically equivalent and different versions of rules."""

    def __init__(self, predicate_vocab: List[str], predicate_arities: Dict[str, int]):
        self.predicate_vocab = predicate_vocab
        self.predicate_arities = predicate_arities

    # ==================== EQUIVALENT TRANSFORMATIONS ====================

    def rename_variables(self, theory: Theory) -> Theory:
        """
        Rename all variables with new IDs (semantically equivalent).

        Example:
            parent(X0, X1) :- child(X1, X0)
            → parent(X5, X3) :- child(X3, X5)
        """
        if not theory:
            return theory

        rule = theory[0]

        # Create mapping from old variables to new variables
        from .logic_structures import get_all_variables
        old_vars = get_all_variables(theory)

        # Generate new random IDs
        new_ids = random.sample(range(100, 100 + len(old_vars) * 2), len(old_vars))
        var_mapping = {old_var: Variable(id=new_id)
                      for old_var, new_id in zip(old_vars, new_ids)}

        # Apply mapping
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
        """
        Randomly reorder body atoms (semantically equivalent).

        Example:
            parent(X0, X1) :- child(X1, X2), adult(X0)
            → parent(X0, X1) :- adult(X0), child(X1, X2)
        """
        if not theory:
            return theory

        rule = theory[0]
        shuffled_body = list(rule.body)
        random.shuffle(shuffled_body)

        return [Rule(head=rule.head, body=shuffled_body)]

    def duplicate_body_atom(self, theory: Theory) -> Theory:
        """
        Duplicate a random body atom (semantically equivalent in logic).

        Example:
            parent(X0, X1) :- child(X1, X0)
            → parent(X0, X1) :- child(X1, X0), child(X1, X0)
        """
        if not theory or not theory[0].body:
            return theory

        rule = theory[0]
        # Pick random atom to duplicate
        atom_to_dup = random.choice(rule.body)
        new_body = list(rule.body) + [atom_to_dup]

        return [Rule(head=rule.head, body=new_body)]

    def apply_equivalent_transform(self, theory: Theory) -> Theory:
        """
        Apply a random equivalent transformation.

        Randomly chooses from:
        - Variable renaming (always safe)
        - Body atom shuffling
        - Atom duplication
        - Combination of the above
        """
        transforms = [
            self.rename_variables,
            self.shuffle_body_atoms,
        ]

        # Sometimes duplicate atoms (less frequently)
        if random.random() < 0.3 and theory and theory[0].body:
            transforms.append(self.duplicate_body_atom)

        # Apply 1-3 transforms
        result = theory
        num_transforms = random.randint(1, min(3, len(transforms)))
        chosen_transforms = random.sample(transforms, num_transforms)

        for transform in chosen_transforms:
            result = transform(result)

        return result

    # ==================== SEMANTIC MODIFICATIONS ====================

    def replace_body_atom(self, theory: Theory) -> Theory:
        """
        Replace one body atom with a different predicate (semantically different).

        Example:
            parent(X0, X1) :- child(X1, X0), adult(X0)
            → parent(X0, X1) :- male(X0), adult(X0)  [replaced child with male]
        """
        if not theory or not theory[0].body:
            return theory

        rule = theory[0]

        # Pick random atom to replace
        idx_to_replace = random.randrange(len(rule.body))
        old_atom = rule.body[idx_to_replace]

        # Choose different predicate with same arity
        same_arity_preds = [p for p in self.predicate_vocab
                           if self.predicate_arities[p] == len(old_atom.args)
                           and p != old_atom.predicate_name]

        if not same_arity_preds:
            # Fallback: change a variable instead
            return self.replace_variable_in_body(theory)

        new_pred = random.choice(same_arity_preds)
        new_atom = Atom(predicate_name=new_pred, args=old_atom.args)

        new_body = list(rule.body)
        new_body[idx_to_replace] = new_atom

        return [Rule(head=rule.head, body=new_body)]

    def replace_variable_in_body(self, theory: Theory) -> Theory:
        """
        Replace a variable in body with a different variable (semantically different).

        Example:
            parent(X0, X1) :- child(X1, X2), adult(X2)
            → parent(X0, X1) :- child(X1, X3), adult(X2)  [broke connection]
        """
        if not theory or not theory[0].body:
            return theory

        rule = theory[0]
        from .logic_structures import get_all_variables

        all_vars = get_all_variables(theory)
        if len(all_vars) < 2:
            return theory

        # Pick a variable to replace (prefer body-only variables)
        head_vars = set(rule.head.args)
        body_only_vars = [v for v in all_vars if v not in head_vars]

        if body_only_vars:
            var_to_replace = random.choice(body_only_vars)
        else:
            # Last resort: pick any body variable
            body_vars = set()
            for atom in rule.body:
                body_vars.update(atom.args)
            if not body_vars:
                return theory
            var_to_replace = random.choice(list(body_vars))

        # Create new variable
        max_id = max(v.id for v in all_vars)
        new_var = Variable(id=max_id + 1)

        # Replace in body only (not head)
        def replace_in_args(args):
            return tuple(new_var if arg == var_to_replace else arg for arg in args)

        new_body = [
            Atom(predicate_name=atom.predicate_name, args=replace_in_args(atom.args))
            for atom in rule.body
        ]

        return [Rule(head=rule.head, body=new_body)]

    def add_extra_atom(self, theory: Theory) -> Theory:
        """
        Add an extra body atom (semantically different - adds constraint).

        Example:
            parent(X0, X1) :- child(X1, X0)
            → parent(X0, X1) :- child(X1, X0), male(X0)
        """
        if not theory:
            return theory

        rule = theory[0]
        from .logic_structures import get_all_variables

        all_vars = get_all_variables(theory)

        # Pick random predicate
        pred_name = random.choice(self.predicate_vocab)
        arity = self.predicate_arities[pred_name]

        # Use existing variables
        if len(all_vars) >= arity:
            atom_vars = random.sample(all_vars, arity)
        else:
            # Create new variables if needed
            atom_vars = list(all_vars)
            max_id = max(v.id for v in all_vars) if all_vars else -1
            for i in range(arity - len(all_vars)):
                atom_vars.append(Variable(id=max_id + i + 1))

        new_atom = Atom(predicate_name=pred_name, args=tuple(atom_vars))
        new_body = list(rule.body) + [new_atom]

        return [Rule(head=rule.head, body=new_body)]

    def remove_body_atom(self, theory: Theory) -> Theory:
        """
        Remove a body atom (semantically different - relaxes constraint).

        Example:
            parent(X0, X1) :- child(X1, X0), adult(X0)
            → parent(X0, X1) :- child(X1, X0)
        """
        if not theory or len(theory[0].body) <= 1:
            # Don't remove if only one atom (would create invalid rule)
            return self.replace_body_atom(theory)

        rule = theory[0]

        # Remove random atom
        idx_to_remove = random.randrange(len(rule.body))
        new_body = [atom for i, atom in enumerate(rule.body) if i != idx_to_remove]

        # Check that head variables are still in body
        head_vars = set(rule.head.args)
        body_vars = set()
        for atom in new_body:
            body_vars.update(atom.args)

        if not head_vars.issubset(body_vars):
            # Would create free variables - abort
            return self.replace_body_atom(theory)

        return [Rule(head=rule.head, body=new_body)]

    def apply_semantic_modification(self, theory: Theory) -> Theory:
        """
        Apply a random semantic modification (creates different rule).

        Randomly chooses from:
        - Replace body atom
        - Replace variable
        - Add extra atom
        - Remove body atom
        """
        modifications = [
            self.replace_body_atom,
            self.replace_variable_in_body,
            self.add_extra_atom,
            self.remove_body_atom,
        ]

        # Apply 1-2 modifications
        result = theory
        num_mods = random.randint(1, 2)

        for _ in range(num_mods):
            mod = random.choice(modifications)
            result = mod(result)

        return result


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning rule embeddings.

    Uses NT-Xent (Normalized Temperature-scaled Cross Entropy) loss from SimCLR.

    Given:
    - Anchor embedding: z_anchor
    - Positive embedding: z_pos (equivalent rule)
    - Negative embeddings: z_neg_i (different rules)

    Loss encourages:
    - similarity(z_anchor, z_pos) to be high
    - similarity(z_anchor, z_neg_i) to be low

    Formula:
        L = -log(exp(sim(z_a, z_p) / τ) / Σ_i exp(sim(z_a, z_i) / τ))

    where τ is temperature parameter.
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor_embeddings: torch.Tensor,
                positive_embeddings: torch.Tensor,
                negative_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            anchor_embeddings: [batch_size, embedding_dim]
            positive_embeddings: [batch_size, embedding_dim]
            negative_embeddings: [batch_size, num_negatives, embedding_dim]

        Returns:
            loss: Scalar contrastive loss
        """
        batch_size = anchor_embeddings.size(0)

        # Normalize embeddings
        anchor_norm = F.normalize(anchor_embeddings, dim=1)
        positive_norm = F.normalize(positive_embeddings, dim=1)
        negative_norm = F.normalize(negative_embeddings, dim=2)

        # Compute similarities
        # Positive similarity: [batch_size]
        pos_sim = torch.sum(anchor_norm * positive_norm, dim=1) / self.temperature

        # Negative similarities: [batch_size, num_negatives]
        neg_sim = torch.bmm(
            negative_norm,  # [batch_size, num_negatives, embedding_dim]
            anchor_norm.unsqueeze(2)  # [batch_size, embedding_dim, 1]
        ).squeeze(2) / self.temperature  # [batch_size, num_negatives]

        # Concatenate positive and negative similarities
        # [batch_size, 1 + num_negatives]
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=logits.device)

        # Cross-entropy loss
        loss = F.cross_entropy(logits, labels)

        return loss


class EncoderPretrainer:
    """Pretrains the GNN encoder using contrastive learning on random rules."""

    def __init__(self,
                 state_encoder: StateEncoder,
                 graph_constructor: GraphConstructor,
                 predicate_vocab: List[str],
                 predicate_arities: Dict[str, int],
                 learning_rate: float = 1e-3,
                 temperature: float = 0.5,
                 num_negatives: int = 4):
        """
        Args:
            state_encoder: GNN encoder to pretrain
            graph_constructor: Converts theories to graphs
            predicate_vocab: List of predicates
            predicate_arities: Mapping from predicate to arity
            learning_rate: Learning rate for pretraining
            temperature: Temperature for contrastive loss
            num_negatives: Number of negative samples per anchor
        """
        self.encoder = state_encoder
        self.graph_constructor = graph_constructor

        self.rule_generator = RandomRuleGenerator(predicate_vocab, predicate_arities)
        self.augmenter = RuleAugmenter(predicate_vocab, predicate_arities)

        self.contrastive_loss = ContrastiveLoss(temperature=temperature)
        self.num_negatives = num_negatives

        # Optimizer for encoder only
        self.optimizer = torch.optim.Adam(state_encoder.parameters(), lr=learning_rate)

    def create_training_batch(self, batch_size: int) -> Tuple[List[Theory], List[Theory], List[List[Theory]]]:
        """
        Create a batch of (anchor, positive, negatives) rule triplets.

        Returns:
            anchors: List of original rules
            positives: List of equivalent rules (same as anchors but augmented)
            negatives: List of lists, where negatives[i] contains semantically different rules
        """
        # Generate anchor rules
        anchors = self.rule_generator.generate_batch(batch_size)

        # Generate positives (equivalent transformations)
        positives = [self.augmenter.apply_equivalent_transform(anchor)
                     for anchor in anchors]

        # Generate negatives (semantic modifications)
        negatives = []
        for anchor in anchors:
            neg_samples = []
            for _ in range(self.num_negatives):
                # Mix of: modified version of anchor + completely different rules
                if random.random() < 0.6:
                    # Modify the anchor
                    neg = self.augmenter.apply_semantic_modification(anchor)
                else:
                    # Completely different rule
                    neg = self.rule_generator.generate_random_rule()
                neg_samples.append(neg)
            negatives.append(neg_samples)

        return anchors, positives, negatives

    def encode_theory(self, theory: Theory) -> torch.Tensor:
        """Encode a theory into an embedding vector."""
        graph_data = self.graph_constructor.theory_to_graph(theory)
        graph_embedding, _ = self.encoder(graph_data)
        return graph_embedding.squeeze(0)  # Remove batch dimension

    def pretrain_step(self, batch_size: int) -> Dict[str, float]:
        """
        Perform one pretraining step.

        Returns:
            metrics: Dictionary with loss and accuracy
        """
        # Create batch
        anchors, positives, negatives = self.create_training_batch(batch_size)

        # Encode all theories
        anchor_embeds = torch.stack([self.encode_theory(a) for a in anchors])
        positive_embeds = torch.stack([self.encode_theory(p) for p in positives])

        # Encode negatives: [batch_size, num_negatives, embedding_dim]
        negative_embeds = []
        for neg_list in negatives:
            neg_embeds_i = torch.stack([self.encode_theory(neg) for neg in neg_list])
            negative_embeds.append(neg_embeds_i)
        negative_embeds = torch.stack(negative_embeds)

        # Compute loss
        loss = self.contrastive_loss(anchor_embeds, positive_embeds, negative_embeds)

        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Compute accuracy (is positive more similar than negatives?)
        with torch.no_grad():
            anchor_norm = F.normalize(anchor_embeds, dim=1)
            positive_norm = F.normalize(positive_embeds, dim=1)
            negative_norm = F.normalize(negative_embeds, dim=2)

            # Positive similarities
            pos_sim = torch.sum(anchor_norm * positive_norm, dim=1)  # [batch_size]

            # Negative similarities
            neg_sim = torch.bmm(
                negative_norm,
                anchor_norm.unsqueeze(2)
            ).squeeze(2)  # [batch_size, num_negatives]

            # Accuracy: positive should be more similar than all negatives
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
        """
        Pretrain the encoder for a number of steps.

        Args:
            num_steps: Number of pretraining steps
            batch_size: Batch size for each step
            verbose: Whether to print progress
            log_interval: Print every N steps

        Returns:
            history: List of metrics for each step
        """
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
        """Save the pretrained encoder weights."""
        torch.save(self.encoder.state_dict(), path)
        print(f"Saved pretrained encoder to {path}")

    def load_pretrained_encoder(self, path: str):
        """Load pretrained encoder weights."""
        self.encoder.load_state_dict(torch.load(path))
        print(f"Loaded pretrained encoder from {path}")


def visualize_augmentations(predicate_vocab: List[str],
                            predicate_arities: Dict[str, int],
                            num_examples: int = 3):
    """
    Utility function to visualize rule augmentations.
    Useful for debugging and understanding transformations.
    """
    print("=" * 80)
    print("RULE AUGMENTATION EXAMPLES")
    print("=" * 80)

    generator = RandomRuleGenerator(predicate_vocab, predicate_arities)
    augmenter = RuleAugmenter(predicate_vocab, predicate_arities)

    for i in range(num_examples):
        print(f"\n--- Example {i+1} ---")

        # Generate original rule
        original = generator.generate_random_rule()
        print(f"\nOriginal Rule:")
        print(f"  {theory_to_string(original)}")

        # Show equivalent transformations
        print(f"\nEquivalent Transformations (semantically same):")

        renamed = augmenter.rename_variables(original)
        print(f"  1. Renamed variables: {theory_to_string(renamed)}")

        shuffled = augmenter.shuffle_body_atoms(original)
        print(f"  2. Shuffled atoms: {theory_to_string(shuffled)}")

        if original[0].body:
            duplicated = augmenter.duplicate_body_atom(original)
            print(f"  3. Duplicated atom: {theory_to_string(duplicated)}")

        # Show semantic modifications
        print(f"\nSemantic Modifications (semantically different):")

        replaced_atom = augmenter.replace_body_atom(original)
        print(f"  1. Replaced atom: {theory_to_string(replaced_atom)}")

        replaced_var = augmenter.replace_variable_in_body(original)
        print(f"  2. Replaced variable: {theory_to_string(replaced_var)}")

        added_atom = augmenter.add_extra_atom(original)
        print(f"  3. Added atom: {theory_to_string(added_atom)}")

        if len(original[0].body) > 1:
            removed_atom = augmenter.remove_body_atom(original)
            print(f"  4. Removed atom: {theory_to_string(removed_atom)}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    # Example usage
    predicate_vocab = ['parent', 'child', 'male', 'female', 'adult', 'sibling', 'ancestor', 'friend']
    predicate_arities = {
        'parent': 2, 'child': 2, 'sibling': 2, 'ancestor': 2, 'friend': 2,
        'male': 1, 'female': 1, 'adult': 1
    }

    # Visualize augmentations
    visualize_augmentations(predicate_vocab, predicate_arities, num_examples=3)
