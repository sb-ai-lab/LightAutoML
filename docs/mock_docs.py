"""A one line summary of the module or program, terminated by a period.

Leave one blank line.  The rest of this docstring should contain an
overall description of the module or program.  Optionally, it may also
contain a brief description of exported classes and functions and/or usage
examples.

    Typical usage example:

        >>> print('something')
        something
        >>> a = MyClass('be', 'or', 'not')

"""

import datetime


class MyClass:
    """Description of class.

    Really do nothing.

    Attributes:
        attr1 (str): Description of `attr1`.
        attr2 (str): Description of `attr2`.

    Args:
        attr1: Description of `attr1`.
        attr2: Description of `attr2`.


    """

    def __init__(self, attr1: str, attr2: str):
        self.attr1 = attr1
        self.attr2 = attr2
        date = datetime.datetime.now()
        print("{}.{}.{} {}:{}:{}".format(date.day, date.month, date.year, date.hour, date.minute, date.second))


# .. toctree::
#     :glob:
#     :maxdepth: 1
#     :caption: Tutorials
#
#     tutorials/tutor_1.ipynb
#     tutorials/tutor_2.ipynb
#     tutorials/tutor_3.ipynb
