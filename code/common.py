from progress.bar import Bar


class ProgressBar(Bar):
    check_tty = False  # for PyCharm, see https://github.com/verigak/progress/issues/50
    suffix = '%(index)d/%(max)d - %(elapsed)ds - ETA: %(eta)ds'
