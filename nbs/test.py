from nbdev.showdoc import show_doc
import sys

if __name__ == '__main__':
    sys.path += ['../flax_experiments']
    from flax_experiments.model import ConvNeXt
    show_doc(ConvNeXt)