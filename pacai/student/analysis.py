"""
Analysis question.
Change these default values to obtain the specified policies through value iteration.
If any question is not possible, return just the constant NOT_POSSIBLE:
```
return NOT_POSSIBLE
```
"""

NOT_POSSIBLE = None

def question2():
    """
    I changed noise to 0, so the agent has no chance of accidentally walking into
    one of the treacherous -100 score pre-terminal states.
    """

    answerDiscount = 0.9
    answerNoise = 0

    return answerDiscount, answerNoise

def question3a():
    """
    I made the discount incredibly steep, so that the distant reward is basically worthless
    to the agent. Noise is 0.01 so there's little risk of diving into the death cliffs, and the
    living reward is -1 to discourage the agent from taking the longer path.
    """

    answerDiscount = 0.2
    answerNoise = 0.01
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward

def question3b():
    """
    Noise is high, making the cliffside especially risky.
    Between the low living reward and reduced discount rate, the agent will avoid taking
    the longer path to the higher reward terminal state.
    """

    answerDiscount = 0.7
    answerNoise = 0.5
    answerLivingReward = -1.5

    return answerDiscount, answerNoise, answerLivingReward

def question3c():
    """
    Living reward is low, making the agent prefer a shorter path.
    Noise is low as well, so the cliffside isn't very risky.
    """

    answerDiscount = 0.9
    answerNoise = 0.01
    answerLivingReward = -1

    return answerDiscount, answerNoise, answerLivingReward

def question3d():
    """
    The noise is very high here, making the cliffside dangerous.
    The small (but still negative) living reward allows the agent to
    take a longer path.
    """

    answerDiscount = 0.9
    answerNoise = 0.5
    answerLivingReward = -0.1

    return answerDiscount, answerNoise, answerLivingReward

def question3e():
    """
    The discount is 1, so values increase an absurd amount through value propagation.
    The noise is large enough for the agent to avoid the cliffside.
    The living reward is extremely high, so the agent has no reason to go to a terminal state.
    """

    answerDiscount = 1
    answerNoise = 0.1
    answerLivingReward = 20

    return answerDiscount, answerNoise, answerLivingReward

def question6():
    """
    When the agent is exploring randomly, the chances of it reaching the other end of the
    map are very slim. When it's exploring from learned policy, it will always try to return to the
    leftmost end to get a safe reward. This is impossible using randomness to encourage
    exploration.
    """

    return NOT_POSSIBLE

if __name__ == '__main__':
    questions = [
        question2,
        question3a,
        question3b,
        question3c,
        question3d,
        question3e,
        question6,
    ]

    print('Answers to analysis questions:')
    for question in questions:
        response = question()
        print('    Question %-10s:\t%s' % (question.__name__, str(response)))
