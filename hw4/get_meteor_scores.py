import subprocess

def new_ref(outfilename):
    all_hyps = [pair.split(' ||| ')[1] for pair in open('data/dev.100best')]
    all_refs = [ref for ref in open('data/dev.ref')]
    num_sents = len(all_hyps) / 100
    outfile = open(outfilename, 'w')
    for s in xrange(0, num_sents):
        hyps_for_one_sent = all_hyps[s * 100:s * 100 + 100]
        ref = all_refs[s]
        for hyp in hyps_for_one_sent:
            outfile.write(ref)
    outfile.close()

def new_hyps(outfilename):
    all_hyps = [pair.split(' ||| ')[1] for pair in open('data/dev.100best')]
    outfile = open(outfilename, 'w')
    for hyp in all_hyps:
        outfile.write(hyp + '\n')
    outfile.close()

def run_meteor(hyp_filename, ref_filename):
    meteor_jar = './meteor-1.4/meteor-1.4.jar'
    shell_command = ['java', '-Xmx1G', '-jar ' + meteor_jar,
                     hyp_filename, ref_filename]
    meteor_stdout = subprocess.check_output(' '.join(shell_command), shell=True)
    print meteor_stdout

if __name__ == '__main__':
    new_ref('data/dev.ref.new')
    new_hyps('data/dev.100best.sents')
    run_meteor('data/dev.100best.sents', 'data/dev.ref.new')
