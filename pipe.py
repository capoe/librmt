from external.soap.soapy.momo import ExtendableNamespace

class Pipe(object):
    def __init__(self, label=""):
        self.label = label
        self.sequence = []
        self.tags = []
        self.hyper = None
        return
    def push(self, fct, tag=""):
        self.sequence.append(fct)
        self.tags.append(tag)
        return
    def apply(self, state, options, logger, stop_at=None, start_at=None, prefix=True):
        if prefix: logger << logger.mb << "Entering pipe '%s' ..." % self.label << logger.endl
        logger.prefix += '[%s] ' % (self.label)
        output = {}
        start_idx = self.tags.index(start_at) if start_at != None else 0
        stop_idx = self.tags.index(stop_at) if stop_at != None else len(self.tags)
        stage_idx = -1
        for fct, tag in zip(self.sequence, self.tags):
            stage_idx += 1
            if stage_idx < start_idx: continue
            if stage_idx >= stop_idx: break
            out = fct(state, options, logger)
            if type(out) == tuple and len(out) == 2: # (state, res)
                output[tag] = out[1]
        logger.prefix = logger.prefix[0:-len(self.label)-3]
        if prefix: logger << logger.mb << "Leaving pipe '%s' ..." % self.label << logger.endl
        return output
    def execute(self, state, options, logger):
        logger << logger.mb << "Entering pipe '%s' ..." % self.label << logger.endl
        logger.prefix += '[%s] ' % (self.label)
        output = []
        for fct, tag in zip(self.sequence, self.tags):
            out = fct(state, options, logger)
            if type(out) == tuple and len(out) == 2: # (state, res)
                res = ExtendableNamespace()
                res.tag = tag
                res.res = out[1]
                output.append(res)
        logger.prefix = logger.prefix[0:-len(self.label)-3]
        logger << logger.mb << "Leaving pipe '%s' ..." % self.label << logger.endl
        return output
    def run(self, state, options, logger):
        logger << logger.mb << "Entering pipe '%s' ..." % self.label << logger.endl
        logger.prefix += '[%s] ' % (self.label)
        output = {}
        for fct, tag in zip(self.sequence, self.tags):
            out = fct(state, options, logger)
            if type(out) == tuple and len(out) == 2: # (state, res)
                if tag != "":
                    output[tag] = out[1]
        logger.prefix = logger.prefix[0:-len(self.label)-3]
        logger << logger.mb << "Leaving pipe '%s' ..." % self.label << logger.endl
        return output
