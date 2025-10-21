"""
Contrastive pre-training for the graph encoder.

Teaches the encoder to distinguish semantically different rules
before GFlowNet training begins.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.logic_structures import (
    Rule, Atom, Variable, apply_add_atom, apply_unify_vars,
    get_valid_variable_pairs, get_all_variables
)
import random
import numpy as np


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for learning rule embeddings.

    Encourages similar embeddings for semantically equivalent rules
    and different embeddings for semantically different rules.
    """

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, anchor, positive, negatives):
        """
        Args:
            anchor: Embedding of original rule [batch_size, embed_dim]
            positive: Embedding of semantically equivalent rule [batch_size, embed_dim]
            negatives: Embeddings of different rules [batch_size, num_negatives, embed_dim]
        """
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=-1)
        positive = F.normalize(positive, dim=-1)
        negatives = F.normalize(negatives, dim=-1)

        # Compute similarities
        pos_sim = torch.sum(anchor * positive, dim=-1) / self.temperature  # [batch_size]

        # Negative similarities
        neg_sim = torch.bmm(
            negatives,  # [batch_size, num_negatives, embed_dim]
            anchor.unsqueeze(-1)  # [batch_size, embed_dim, 1]
        ).squeeze(-1) / self.temperature  # [batch_size, num_negatives]

        # Contrastive loss: maximize pos_sim, minimize neg_sim
        # Using InfoNCE loss
        logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1+num_negatives]
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)

        loss = F.cross_entropy(logits, labels)

        return loss


class RuleAugmenter:
    """Generate augmented versions of rules for contrastive learning."""

    @staticmethod
    def variable_renaming(theory):
        """
        Create semantically equivalent rule with renamed variables.
        This should produce SIMILAR embeddings.
        """
        rule = theory[0]

        # Get all unique variables
        variables = get_all_variables(theory)
        var_ids = [v.id for v in variables]

        # Create random mapping to new IDs
        offset = random.randint(100, 500)
        var_map = {old_id: old_id + offset for old_id in var_ids}

        # Apply mapping
        def rename_atom(atom):
            new_args = tuple(Variable(id=var_map[v.id]) for v in atom.args)
            return Atom(predicate_name=atom.predicate_name, args=new_args)

        new_head = rename_atom(rule.head)
        new_body = tuple(rename_atom(atom) for atom in rule.body)

        return [Rule(head=new_head, body=new_body)]

    @staticmethod
    def predicate_reordering(theory):
        """
        Reorder predicates in body (if more than 1).
        This should produce SIMILAR embeddings.
        """
        rule = theory[0]

        if len(rule.body) <= 1:
            return theory

        # Shuffle body atoms
        new_body = list(rule.body)
        random.shuffle(new_body)

        return [Rule(head=rule.head, body=tuple(new_body))]

    @staticmethod
    def variable_connection_change(theory, predicate_vocab, predicate_arities):
        """
        Change variable connections to create semantically DIFFERENT rule.
        This should produce DIFFERENT embeddings.

        Example: parent(X,Z), parent(Z,Y) -> parent(X,Z), parent(Y,Z)
        (chain becomes convergent)
        """
        rule = theory[0]

        if len(rule.body) < 2:
            # Add a disconnected atom for difference
            max_var_id = max([v.id for v in get_all_variables(theory)], default=-1)
            pred_name = random.choice(predicate_vocab)
            arity = predicate_arities[pred_name]

            try:
                new_theory, _ = apply_add_atom(theory, pred_name, arity, max_var_id)
                return new_theory
            except:
                return theory

        # Pick two atoms and swap some arguments
        body_list = list(rule.body)
        idx1, idx2 = random.sample(range(len(body_list)), 2)

        atom1, atom2 = body_list[idx1], body_list[idx2]

        # Swap last argument of atom1 with first argument of atom2
        if len(atom1.args) > 0 and len(atom2.args) > 0:
            args1 = list(atom1.args)
            args2 = list(atom2.args)

            args1[-1], args2[0] = args2[0], args1[-1]

            body_list[idx1] = Atom(predicate_name=atom1.predicate_name, args=tuple(args1))
            body_list[idx2] = Atom(predicate_name=atom2.predicate_name, args=tuple(args2))

            return [Rule(head=rule.head, body=tuple(body_list))]

        return theory

    @staticmethod
    def add_disconnected_atom(theory, predicate_vocab, predicate_arities):
        """
        Add a disconnected atom to create a DIFFERENT rule.
        """
        max_var_id = max([v.id for v in get_all_variables(theory)], default=-1)
        pred_name = random.choice(predicate_vocab)
        arity = predicate_arities[pred_name]

        # Create atom with completely new variables
        new_var_start = max_var_id + 10
        new_args = tuple(Variable(id=new_var_start + i) for i in range(arity))
        new_atom = Atom(predicate_name=pred_name, args=new_args)

        rule = theory[0]
        new_body = tuple(rule.body) + (new_atom,)

        return [Rule(head=rule.head, body=new_body)]


class ContrastivePreTrainer:
    """Pre-train the encoder using contrastive learning."""

    def __init__(self, state_encoder, graph_constructor, predicate_vocab, predicate_arities):
        self.state_encoder = state_encoder
        self.graph_constructor = graph_constructor
        self.predicate_vocab = predicate_vocab
        self.predicate_arities = predicate_arities

        self.augmenter = RuleAugmenter()
        self.criterion = ContrastiveLoss(temperature=0.5)
        self.optimizer = torch.optim.Adam(state_encoder.parameters(), lr=1e-3)

    def generate_training_batch(self, base_rules, num_negatives=5):
        """
        Generate a batch of contrastive examples.

        Args:
            base_rules: List of base theories to augment
            num_negatives: Number of negative examples per anchor

        Returns:
            anchors, positives, negatives (each as embeddings)
        """
        anchors = []
        positives = []
        negatives_list = []

        for theory in base_rules:
            # Anchor: original rule
            anchor_emb = self.get_embedding(theory)
            anchors.append(anchor_emb)

            # Positive: semantically equivalent (renamed variables or reordered)
            if random.random() > 0.5:
                pos_theory = self.augmenter.variable_renaming(theory)
            else:
                pos_theory = self.augmenter.predicate_reordering(theory)

            pos_emb = self.get_embedding(pos_theory)
            positives.append(pos_emb)

            # Negatives: semantically different
            neg_embs = []
            for _ in range(num_negatives):
                if random.random() > 0.5:
                    neg_theory = self.augmenter.variable_connection_change(
                        theory, self.predicate_vocab, self.predicate_arities
                    )
                else:
                    neg_theory = self.augmenter.add_disconnected_atom(
                        theory, self.predicate_vocab, self.predicate_arities
                    )

                neg_emb = self.get_embedding(neg_theory)
                neg_embs.append(neg_emb)

            negatives_list.append(torch.stack(neg_embs))

        anchors = torch.stack(anchors)
        positives = torch.stack(positives)
        negatives = torch.stack(negatives_list)

        return anchors, positives, negatives

    def get_embedding(self, theory):
        """Get embedding for a theory."""
        graph_data = self.graph_constructor.theory_to_graph(theory)
        state_embedding, _ = self.state_encoder(graph_data)
        return state_embedding.squeeze(0)

    def train_epoch(self, base_rules, num_negatives=5):
        """Train for one epoch."""
        self.state_encoder.train()

        anchors, positives, negatives = self.generate_training_batch(
            base_rules, num_negatives=num_negatives
        )

        loss = self.criterion(anchors, positives, negatives)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def pretrain(self, base_rules, num_epochs=100, verbose=True):
        """
        Pre-train the encoder.

        Args:
            base_rules: List of base theories to use for training
            num_epochs: Number of training epochs
            verbose: Print progress
        """
        losses = []

        if verbose:
            print("=" * 80)
            print("CONTRASTIVE PRE-TRAINING")
            print("=" * 80)
            print(f"\nTraining for {num_epochs} epochs with {len(base_rules)} base rules")

        for epoch in range(num_epochs):
            loss = self.train_epoch(base_rules)
            losses.append(loss)

            if verbose and epoch % 10 == 0:
                print(f"Epoch {epoch:3d}/{num_epochs}: Loss = {loss:.4f}")

        if verbose:
            print(f"\nPre-training complete!")
            print(f"Final loss: {losses[-1]:.4f}")
            print(f"Initial loss: {losses[0]:.4f}")
            print(f"Improvement: {losses[0] - losses[-1]:.4f}")

        return losses


def generate_base_rules(predicate_vocab, predicate_arities, num_rules=50):
    """
    Generate a diverse set of base rules for pre-training.
    """
    from src.logic_structures import get_initial_state

    base_rules = []

    # Start with simple initial states
    for pred in predicate_vocab:
        arity = predicate_arities[pred]
        initial = get_initial_state(pred, arity)
        base_rules.append(initial)

    # Generate rules of different lengths
    for _ in range(num_rules):
        pred = random.choice(predicate_vocab)
        arity = predicate_arities[pred]
        theory = get_initial_state(pred, arity)

        # Add 1-3 body atoms
        num_atoms = random.randint(1, 3)
        max_var_id = max([v.id for v in get_all_variables(theory)], default=-1)

        for _ in range(num_atoms):
            pred_name = random.choice(predicate_vocab)
            pred_arity = predicate_arities[pred_name]

            try:
                theory, max_var_id = apply_add_atom(theory, pred_name, pred_arity, max_var_id)
            except:
                break

        base_rules.append(theory)

    return base_rules


def main():
    """Example usage."""
    print("Contrastive Pre-training for Graph Encoder")
    print("\nThis script pre-trains the encoder to distinguish between")
    print("semantically different rules before GFlowNet training.")
    print("\nSee Demo_ILP.ipynb for integration.")


if __name__ == "__main__":
    main()
