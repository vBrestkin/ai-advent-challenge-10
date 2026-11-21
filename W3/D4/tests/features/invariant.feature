Feature: Database rule for stack proposals
  Scenario: A conflicting proposal is blocked before rendering
    Given a conversation with SQLite as the approved database
    When the model proposes PostgreSQL
    Then the user receives a refusal and the rule remains SQLite
