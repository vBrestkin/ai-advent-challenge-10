Feature: Independent memory layers
  Scenario: Clear one layer in an independent copy
    Given three isolated pieces of knowledge
    When a copy loses its working memory
    Then only the working fact is removed from the model context
