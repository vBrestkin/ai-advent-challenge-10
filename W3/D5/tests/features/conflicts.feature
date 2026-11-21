Feature: Current project facts
  Scenario: The newer deadline wins and the earlier deadline stays in the audit only
    Given a project with a recorded deadline
    When the deadline is changed in chat
    Then the model sees only the new deadline and the audit records both values
