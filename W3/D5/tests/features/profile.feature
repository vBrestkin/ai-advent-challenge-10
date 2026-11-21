Feature: Persistent user preferences
  Scenario: An explicit chat preference survives restart
    Given two separate conversations
    When the user updates one profile in chat
    Then after reopening only that conversation uses the new preference
