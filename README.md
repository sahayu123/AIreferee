# AIreferee

Football incident review with separate general-foul and handball specialists.

## Dataset

The handball-classification dataset used by this project is publicly available
on Kaggle: [Football Handball vs No Handball](https://www.kaggle.com/datasets/sahayuraja/football-handball-vs-no-handball).

We created the dataset curation and annotations from scratch for AIreferee. Our
work includes selecting the incidents, creating the temporal clips, labeling
each example as `handball` or `not_handball`, extracting the frame sequences,
and organizing the metadata and class structure. The current release contains
286 labeled examples: 86 handball and 200 not-handball. The underlying match
and broadcast footage remains the property of its respective publishers and
rights holders.

## GPU review application

The professional Apple GPU localhost interface and Python inference backend are
in [`vair-gpu-ui/`](vair-gpu-ui/README.md). It runs the image detector on every
decoded video frame, tracks players and the ball, evaluates pose/contact
evidence, and adds the Handball Detection Project's trajectory, arm-angle, and
high-confidence ball-to-arm proximity rules.
