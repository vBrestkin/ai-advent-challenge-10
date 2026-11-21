Feature: Task lifecycle
  Scenario: Restoring a paused task preserves the expected step
    Given a task with an approved plan
    When the task is paused and the application reopened
    Then the task resumes at the same step with its approved plan
