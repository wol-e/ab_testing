# Sequential AB testing

## General Notes

Pros:
- Depending on design, allows evaluating the data at multiple times during 
the testing phase, possibly at any time
- Possibility to stop experiments early for success or futility in a principled
way
- On average reduces runtime of experiments when there is a real effect.
The literature suggest improved runtimes up to 50% or even 80%
- The problem of peeking can be entirely mitigated which makes using a platform
potentially easier for experimenters, but this depends on the approach

Cons:
- Generally lower power as compared to fixed horizon test (this needs more details)
- When no effect is measurable for a test, runtimes are longer than a fixed horizon test
- It appears so far that there are no out of the box solutions in python (this is where R wins, they have it)
- More complicated implementation on backend and frontend, depending on chosen approach. E.g. sequential
testing often requires a pre-determined maximum sample size or registering looks at the data by the experimenter.