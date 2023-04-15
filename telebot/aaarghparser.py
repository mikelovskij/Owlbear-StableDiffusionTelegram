import argparse
import sys


# a slightly modified argumentparser that does not exit stupidly on error but raises an exception instead
class MyParser(argparse.ArgumentParser):

    def exit(self, status=0, message=None):
        if message:
            self._print_message(message, sys.stderr)
        raise ValueError(message)


# Add arguments
def init_parser():
    parser = MyParser(prog=' ', exit_on_error=False)

    parser.add_argument('-d', '--dream', nargs='*', dest='dream')
    parser.add_argument('-S', '--Seed', type=int, dest='seed', help='Use specified seed instead of random')

    parser.add_argument('-P', '--Portrait', action='count', default=0, dest='portrait', help='Increases the H/W ratio')
    parser.add_argument('-L', '--Landscape', action='count', default=0, dest='landscape', help='Decreases the H/W ratio')

    parser.add_argument('-l', '--load', dest='load')

    parser.add_argument('-H', '--height', type=int, dest='width', default=640)
    parser.add_argument('-W', '--width', type=int, dest='height', default=640)
    parser.add_argument('-s', '--steps', type=int, dest='steps', default=30, help='Number of diffusion steps')
    parser.add_argument('-c', '--cfg_scale', type=float, dest='cfg_scale', default=6,
                        help='How hard it tries to follow the prompt, kind of, from 1 to 20')
    parser.add_argument('-DS', '--Denoising_Strength', type=float, dest='denoising_strength', default=0.6,
                        help='Strength of denoising in IMG2IMG')

    parser.add_argument('-o', '--owlbearify', type=int, dest='owlbearify', default=0)
    return parser
