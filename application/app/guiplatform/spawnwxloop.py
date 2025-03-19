# ruff: noqa: ANN001, N802, N806
# mypy: disable-error-code="no-untyped-def"

# Copyright (c) 2008, Carnegie Mellon University.
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:

# 1. Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
# 2. Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
# 3. Neither the name of Carnegie Mellon University nor the names of
#    other contributors may be used to endorse or promote products
#    derived from this software without specific prior written
#    permission.

# THIS SOFTWARE IS PROVIDED BY THE AUTHORS "AS IS" AND ANY EXPRESS OR
# IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES
# OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
# IN NO EVENT SHALL THE AUTHORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
# NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
# THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# (This is the Modified BSD License.  See also
# http://www.opensource.org/licenses/bsd-license.php )

"""
Modified version of `spawnWxLoop()` defined in Panda3d 1.10.

(It was outdated.)
"""

from typing import cast

import wx

from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import ConfigVariableBool, ConfigVariableDouble
from panda3d.direct import init_app_for_gui


def spawnWxLoop(wx_app: wx.App, base: ShowBase) -> None:
    """
    Call this method to hand the main loop over to wxPython.

    This sets up a wxTimer callback so that Panda still gets
    updated, but wxPython owns the main loop (which seems to make
    it happier than the other way around).
    """
    if base.wxAppCreated:
        # Don't do this twice.
        return

    init_app_for_gui()

    base.wxApp = wx_app

    base.wxTimer = None

    if ConfigVariableBool('wx-main-loop', default_value=True):
        # Put wxPython in charge of the main loop.  It really
        # seems to like this better; some features of wx don't
        # work properly unless this is true.

        # Set a timer to run the Panda frame 60 times per second.
        wxFrameRate = ConfigVariableDouble('wx-frame-rate', 60.0)
        base.wxTimer = wx.Timer(wx_app)
        wx_app.Bind(wx.EVT_TIMER, base._ShowBase__wxTimerCallback)  # noqa: SLF001
        base.wxTimer.Start(
            round(1000.0 / wxFrameRate.getValue())
        )  # Fixed in https://github.com/panda3d/panda3d/blob/4f9092d568bc499e6f26241ee68c5e1a10eb470c/direct/src/showbase/ShowBase.py#L3240

        # wx is now the main loop, not us any more.
        base.run = base.wxRun
        base.taskMgr.run = base.wxRun
        # builtins.run = base.wxRun
        if base.appRunner:
            base.appRunner.run = base.wxRun

    else:
        # Leave Panda in charge of the main loop.  This is
        # friendlier for IDE's and interactive editing in general.
        def wxLoop(task: Task) -> int:
            # First we need to ensure that the OS message queue is
            # processed.
            base.wxApp.Yield()

            # Now do all the wxPython events waiting on this frame.
            # CHECK is supposed to fail here https://docs.wxpython.org/wx.EvtHandler.html#wx.EvtHandler.ProcessPendingEvents
            base.wxApp.ProcessPendingEvents()

            return cast(int, task.again)

        base.taskMgr.add(wxLoop, 'wxLoop')

    base.wxAppCreated = True
