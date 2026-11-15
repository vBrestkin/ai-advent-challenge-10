Feature: Persistent full conversation context
  Scenario: Restore a conversation including tool calls after a restart
    Given a saved conversation with an order lookup
    When a new application instance continues that conversation
    Then the model receives the exact saved protocol and the order amount
    And the browser history includes both successful turns
