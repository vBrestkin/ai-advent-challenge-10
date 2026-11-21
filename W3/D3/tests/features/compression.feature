Feature: Safe persistent summary memory
  Scenario: Compress an old prefix without losing the archive
    Given twelve complete conversation messages
    When summary memory is updated keeping six recent messages
    Then the request contains the summary and six original messages
    And the complete twelve-message archive remains unchanged
