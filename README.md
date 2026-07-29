# AIreferee

Football incident review with separate general-foul and handball specialists.

## GPU review application

The professional Apple GPU localhost interface and Python inference backend are
in [`vair-gpu-ui/`](vair-gpu-ui/README.md). It runs the image detector on every
decoded video frame, tracks players and the ball, evaluates pose/contact
evidence, and adds the Handball Detection Project's trajectory, arm-angle, and
high-confidence ball-to-arm proximity rules.
