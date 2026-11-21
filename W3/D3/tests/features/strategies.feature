Feature: Context strategies and immutable checkpoints
  Scenario: Two branches evolve independently from one checkpoint
    Given a persisted dialogue with a checkpoint
    When two branches change their own budget
    Then each branch restores its own facts and the source stays unchanged
