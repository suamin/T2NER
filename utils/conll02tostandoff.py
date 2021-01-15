#!/usr/bin/env python

# Script to convert a CoNLL 2002-flavored BIO-formatted entity-tagged
# file into BioNLP ST-flavored standoff and a reconstruction of the
# original text.



import codecs
import os
import re
import sys

INPUT_ENCODING = "Latin-1"
OUTPUT_ENCODING = "UTF-8"

output_directory = None


def quote(s):
    return s in ('"', )


def space(t1, t2, quote_count=None):
    # Helper for reconstructing sentence text. Given the text of two
    # consecutive tokens, returns a heuristic estimate of whether a
    # space character should be placed between them.

    if re.match(r'^[\(]$', t1):
        return False
    if re.match(r'^[.,\)\?\!]$', t2):
        return False
    if quote(t1) and quote_count is not None and quote_count % 2 == 1:
        return False
    if quote(t2) and quote_count is not None and quote_count % 2 == 1:
        return False
    return True


def tagstr(start, end, ttype, idnum, text):
    # sanity checks
    assert '\n' not in text, "ERROR: newline in entity '%s'" % (text)
    assert text == text.strip(), "ERROR: tagged span contains extra whitespace: '%s'" % (text)
    return "T%d\t%s %d %d\t%s" % (idnum, ttype, start, end, text)


def output(infn, docnum, sentences):
    global output_directory

    if output_directory is None:
        txtout = sys.stdout
        soout = sys.stdout
    else:
        outfn = os.path.join(
            output_directory,
            os.path.basename(infn) +
            '-doc-' +
            str(docnum))
        txtout = codecs.open(outfn + '.txt', 'w', encoding=OUTPUT_ENCODING)
        soout = codecs.open(outfn + '.ann', 'w', encoding=OUTPUT_ENCODING)

    offset, idnum = 0, 1

    doctext = ""

    for si, sentence in enumerate(sentences):

        prev_token = None
        curr_start, curr_type = None, None
        quote_count = 0

        for token, ttag, ttype in sentence:

            if curr_type is not None and (ttag != "I" or ttype != curr_type):
                # a previously started tagged sequence does not
                # continue into this position.
                print(tagstr(
                    curr_start, offset, curr_type, idnum, doctext[curr_start:offset]), file=soout)
                idnum += 1
                curr_start, curr_type = None, None

            if prev_token is not None and space(
                    prev_token, token, quote_count):
                doctext = doctext + ' '
                offset += 1

            if curr_type is None and ttag != "O":
                # a new tagged sequence begins here
                curr_start, curr_type = offset, ttype

            doctext = doctext + token
            offset += len(token)

            if quote(token):
                quote_count += 1

            prev_token = token

        # leftovers?
        if curr_type is not None:
            print(tagstr(
                curr_start, offset, curr_type, idnum, doctext[curr_start:offset]), file=soout)
            idnum += 1

        if si + 1 != len(sentences):
            doctext = doctext + '\n'
            offset += 1

    print(doctext, file=txtout)


def process(fn):
    docnum = 1
    sentences = []

    with open(fn, encoding="utf-8", errors="ignore") as f:

        # store (token, BIO-tag, type) triples for sentence
        current = []

        lines = f.readlines()

        for ln, l in enumerate(lines):
            l = l.strip()
            if not l:
                sentences.append(current)
                current = []
                continue
            token, temp = l.split("\t")
            ttag, ttype = temp[0], temp[2:]

            current.append((token, ttag, ttype))

    if len(sentences) > 0:
        output(fn, docnum, sentences)


def main(argv):
    global reference_directory, output_directory

    # (clumsy arg parsing, sorry)

    # Take a mandatory "-d" arg that tells us where to find the original,
    # unsegmented and untagged reference files.

    if len(argv) < 3 or argv[1] != "-d":
        print("USAGE:", argv[0], "-d REF-DIR [-o OUT-DIR] (FILES|DIR)", file=sys.stderr)
        return 1

    reference_directory = argv[2]

    # Take an optional "-o" arg specifying an output directory for the results

    output_directory = None
    filenames = argv[3:]
    if len(argv) > 4 and argv[3] == "-o":
        output_directory = argv[4]
        print("Writing output to %s" % output_directory, file=sys.stderr)
        filenames = argv[5:]

    # special case: if we only have a single file in input and it specifies
    # a directory, process all files in that directory
    input_directory = None
    if len(filenames) == 1 and os.path.isdir(filenames[0]):
        input_directory = filenames[0]
        filenames = [os.path.join(input_directory, fn)
                     for fn in os.listdir(input_directory)]
        print("Processing %d files in %s ..." % (
            len(filenames), input_directory), file=sys.stderr)

    fail_count = 0
    for fn in filenames:
        try:
            process(fn)
        except Exception as e:
            print("Error processing %s: %s" % (fn, e), file=sys.stderr)
            fail_count += 1

            # if we're storing output on disk, remove the output file
            # to avoid having partially-written data
            ofn = output_filename(fn)
            try:
                os.remove(ofn)
            except BaseException:
                # never mind if that fails
                pass

    if fail_count > 0:
        print("""
##############################################################################
#
# WARNING: error in processing %d/%d files, output is incomplete!
#
##############################################################################
""" % (fail_count, len(filenames)), file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))