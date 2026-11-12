"""Bounded onset geometry and repeated project-reader execution."""


def census(onsets=(0, 1, 2)):
    if tuple(onsets) != (0, 1, 2):
        raise ValueError('This assay qualifies onsets0,1,2 only')
    suffixes = sum(28 * (4-k) for k in onsets)
    transitions = 112 + 8 * suffixes  # seven native E variants, one stackedT
    acts = 8 * len(onsets) + sum(4-k for k in onsets)
    return dict(loads=1, encodes=0, noise_draws=0, native_transitions=transitions,
                acts=acts, receipts=transitions+acts, images=1+8*len(onsets))


def run_suffix(nav, *, on_project=None, on_boundary=None):
    while nav.point['logical_step'] < 4:
        if nav.point['phase'] == 'project':
            if on_boundary is not None:
                on_boundary(nav)
            step = nav.point['logical_step']
            receipt = nav.step()
            if on_project is not None:
                on_project(nav, step, receipt)
        else:
            nav.step()
    if on_boundary is not None:
        on_boundary(nav)
