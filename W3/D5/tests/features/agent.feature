Feature: Agent selects and executes tools
  Scenario Outline: Answer using the required number of tool rounds
    Given a model that requests <rounds> tool rounds
    When the agent receives a question
    Then it returns a final answer after <rounds> tool rounds
    Examples:
      | rounds |
      | 0 |
      | 1 |
      | 2 |
