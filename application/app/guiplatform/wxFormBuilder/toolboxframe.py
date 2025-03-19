# -*- coding: utf-8 -*-

###########################################################################
## Python code generated with wxFormBuilder (version 4.1.0-3-g43bf300c)
## http://www.wxformbuilder.org/
##
## PLEASE DO *NOT* EDIT THIS FILE!
###########################################################################

import wx
import wx.xrc

###########################################################################
## Class ToolBoxFrame
###########################################################################

class ToolBoxFrame ( wx.Frame ):

	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = wx.EmptyString, pos = wx.DefaultPosition, size = wx.Size( 451,604 ), style = wx.CAPTION|wx.CLOSE_BOX|wx.MINIMIZE_BOX|wx.SYSTEM_MENU|wx.TAB_TRAVERSAL )

		self.SetSizeHints( wx.DefaultSize, wx.DefaultSize )

		bSizer = wx.BoxSizer( wx.VERTICAL )


		bSizer.Add( ( 0, 5), 1, wx.EXPAND, 5 )

		sbSizer1 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"Each Navigation Window" ), wx.VERTICAL )

		self.m_staticText2 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Target window:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText2.Wrap( -1 )

		sbSizer1.Add( self.m_staticText2, 0, wx.ALL, 5 )

		m_choice_target_windowChoices = []
		self.m_choice_target_window = wx.Choice( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, m_choice_target_windowChoices, wx.CB_SORT )
		self.m_choice_target_window.SetSelection( 0 )
		self.m_choice_target_window.Enable( False )

		sbSizer1.Add( self.m_choice_target_window, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_staticText6 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Visual size:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText6.Wrap( -1 )

		sbSizer1.Add( self.m_staticText6, 0, wx.ALL, 5 )

		fgSizer1 = wx.FlexGridSizer( 0, 3, 0, 0 )
		fgSizer1.SetFlexibleDirection( wx.BOTH )
		fgSizer1.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.m_spinCtrlDouble_visualSize_mantissa = wx.SpinCtrlDouble( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER, 1e-14, 100, 1, 1 )
		self.m_spinCtrlDouble_visualSize_mantissa.SetDigits( 14 )
		fgSizer1.Add( self.m_spinCtrlDouble_visualSize_mantissa, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_staticText3 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"x10^", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText3.Wrap( -1 )

		fgSizer1.Add( self.m_staticText3, 0, wx.ALIGN_CENTER|wx.ALL, 5 )

		self.m_spinCtrl_visualSize_exponent = wx.SpinCtrl( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER, -324, 308, 0 )
		fgSizer1.Add( self.m_spinCtrl_visualSize_exponent, 0, wx.ALL, 5 )


		sbSizer1.Add( fgSizer1, 1, wx.EXPAND, 5 )

		self.m_staticText8 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Shift speed:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText8.Wrap( -1 )

		sbSizer1.Add( self.m_staticText8, 0, wx.ALL, 5 )

		fgSizer3 = wx.FlexGridSizer( 0, 3, 0, 0 )
		fgSizer3.SetFlexibleDirection( wx.BOTH )
		fgSizer3.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.m_spinCtrlDouble_shiftSpeed_mantissa = wx.SpinCtrlDouble( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER, 1e-14, 100, 1, 1 )
		self.m_spinCtrlDouble_shiftSpeed_mantissa.SetDigits( 14 )
		fgSizer3.Add( self.m_spinCtrlDouble_shiftSpeed_mantissa, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_staticText31 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"x10^", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText31.Wrap( -1 )

		fgSizer3.Add( self.m_staticText31, 0, wx.ALIGN_CENTER|wx.ALL, 5 )

		self.m_spinCtrl_shiftSpeed_exponent = wx.SpinCtrl( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER, -324, 308, 0 )
		fgSizer3.Add( self.m_spinCtrl_shiftSpeed_exponent, 0, wx.ALL, 5 )


		sbSizer1.Add( fgSizer3, 1, wx.EXPAND, 5 )

		self.m_staticText10 = wx.StaticText( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Rotation speed:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText10.Wrap( -1 )

		sbSizer1.Add( self.m_staticText10, 0, wx.ALL, 5 )

		fgSizer10 = wx.FlexGridSizer( 0, 2, 0, 0 )
		fgSizer10.AddGrowableCol( 0 )
		fgSizer10.SetFlexibleDirection( wx.BOTH )
		fgSizer10.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.m_slider_rotationSpeed = wx.Slider( sbSizer1.GetStaticBox(), wx.ID_ANY, 50, 0, 90, wx.DefaultPosition, wx.DefaultSize, wx.SL_HORIZONTAL )
		fgSizer10.Add( self.m_slider_rotationSpeed, 1, wx.ALL|wx.EXPAND, 5 )

		self.m_spinCtrlDouble_rotationSpeed = wx.SpinCtrlDouble( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER, 1e-14, 90, 1, 10 )
		self.m_spinCtrlDouble_rotationSpeed.SetDigits( 2 )
		fgSizer10.Add( self.m_spinCtrlDouble_rotationSpeed, 0, wx.ALL, 5 )


		sbSizer1.Add( fgSizer10, 1, wx.EXPAND, 5 )

		fgSizer5 = wx.FlexGridSizer( 0, 3, 0, 0 )
		fgSizer5.AddGrowableCol( 1 )
		fgSizer5.SetFlexibleDirection( wx.BOTH )
		fgSizer5.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.m_button_saveScreeshot = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Screenshot", wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer5.Add( self.m_button_saveScreeshot, 0, wx.ALL, 5 )

		self.m_textCtrl_screenshotPrefx = wx.TextCtrl( sbSizer1.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer5.Add( self.m_textCtrl_screenshotPrefx, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_button_chooseScreenshotFolder = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Browse", wx.DefaultPosition, wx.DefaultSize, wx.BU_EXACTFIT )
		fgSizer5.Add( self.m_button_chooseScreenshotFolder, 0, wx.ALL, 5 )


		sbSizer1.Add( fgSizer5, 1, wx.EXPAND, 5 )

		fgSizer6 = wx.FlexGridSizer( 0, 4, 0, 0 )
		fgSizer6.SetFlexibleDirection( wx.BOTH )
		fgSizer6.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.m_button_help = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Help", wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer6.Add( self.m_button_help, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )

		self.m_button_nearestNeighbor = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Nearest neighbor", wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer6.Add( self.m_button_nearestNeighbor, 0, wx.ALL, 5 )

		self.m_button_display = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, u"Display", wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer6.Add( self.m_button_display, 0, wx.ALL, 5 )

		self.m_button_newScene = wx.Button( sbSizer1.GetStaticBox(), wx.ID_ANY, u"New Scene", wx.DefaultPosition, wx.DefaultSize, 0 )
		fgSizer6.Add( self.m_button_newScene, 0, wx.ALL, 5 )


		sbSizer1.Add( fgSizer6, 1, wx.EXPAND, 5 )


		bSizer.Add( sbSizer1, 1, wx.EXPAND, 5 )


		bSizer.Add( ( 0, 5), 1, wx.EXPAND, 5 )

		sbSizer2 = wx.StaticBoxSizer( wx.StaticBox( self, wx.ID_ANY, u"All Navigation Windows" ), wx.VERTICAL )

		self.m_staticText4 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"Animation interval (secs):", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText4.Wrap( -1 )

		sbSizer2.Add( self.m_staticText4, 0, wx.ALL, 5 )

		self.m_spinCtrlDouble_animationInterval = wx.SpinCtrlDouble( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS, 0.1, 100, 0, 1 )
		self.m_spinCtrlDouble_animationInterval.SetDigits( 1 )
		sbSizer2.Add( self.m_spinCtrlDouble_animationInterval, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_staticText5 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"Number of points:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText5.Wrap( -1 )

		sbSizer2.Add( self.m_staticText5, 0, wx.ALL, 5 )

		self.m_spinCtrlDouble_numberOfPoints = wx.SpinCtrlDouble( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER, 1, 100000, 0, 1000 )
		self.m_spinCtrlDouble_numberOfPoints.SetDigits( 0 )
		sbSizer2.Add( self.m_spinCtrlDouble_numberOfPoints, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_staticText7 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"Neighborhood radius:", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText7.Wrap( -1 )

		sbSizer2.Add( self.m_staticText7, 0, wx.ALL, 5 )

		fgSizer2 = wx.FlexGridSizer( 0, 3, 0, 0 )
		fgSizer2.SetFlexibleDirection( wx.BOTH )
		fgSizer2.SetNonFlexibleGrowMode( wx.FLEX_GROWMODE_SPECIFIED )

		self.m_spinCtrlDouble_neighborhoodRadius_mantissa = wx.SpinCtrlDouble( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER, 1e-14, 100, 1, 1 )
		self.m_spinCtrlDouble_neighborhoodRadius_mantissa.SetDigits( 14 )
		fgSizer2.Add( self.m_spinCtrlDouble_neighborhoodRadius_mantissa, 0, wx.ALL|wx.EXPAND, 5 )

		self.m_staticText9 = wx.StaticText( sbSizer2.GetStaticBox(), wx.ID_ANY, u"x10^", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText9.Wrap( -1 )

		fgSizer2.Add( self.m_staticText9, 0, wx.ALIGN_CENTER|wx.ALL, 5 )

		self.m_spinCtrl_neighborhoodRadius_exponent = wx.SpinCtrl( sbSizer2.GetStaticBox(), wx.ID_ANY, wx.EmptyString, wx.DefaultPosition, wx.DefaultSize, wx.SP_ARROW_KEYS|wx.TE_PROCESS_ENTER, -324, 308, 0 )
		fgSizer2.Add( self.m_spinCtrl_neighborhoodRadius_exponent, 0, wx.ALL, 5 )


		sbSizer2.Add( fgSizer2, 1, wx.EXPAND, 5 )

		self.m_checkBox_shared_neighborhood = wx.CheckBox( sbSizer2.GetStaticBox(), wx.ID_ANY, u"Shared neighborhood", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_checkBox_shared_neighborhood.Hide()

		sbSizer2.Add( self.m_checkBox_shared_neighborhood, 0, wx.ALL, 5 )


		bSizer.Add( sbSizer2, 1, wx.EXPAND, 5 )


		self.SetSizer( bSizer )
		self.Layout()

		self.Centre( wx.BOTH )

		# Connect Events
		self.m_spinCtrlDouble_visualSize_mantissa.Bind( wx.EVT_SPINCTRLDOUBLE, self.m_spinCtrlDouble_visualSize_mantissaOnUpdate )
		self.m_spinCtrlDouble_visualSize_mantissa.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrlDouble_visualSize_mantissaOnUpdate )
		self.m_spinCtrl_visualSize_exponent.Bind( wx.EVT_SPINCTRL, self.m_spinCtrl_visualSize_exponentOnUpdate )
		self.m_spinCtrl_visualSize_exponent.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrl_visualSize_exponentOnUpdate )
		self.m_spinCtrlDouble_shiftSpeed_mantissa.Bind( wx.EVT_SPINCTRLDOUBLE, self.m_spinCtrlDouble_shiftSpeed_mantissaOnUpdate )
		self.m_spinCtrlDouble_shiftSpeed_mantissa.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrlDouble_shiftSpeed_mantissaOnUpdate )
		self.m_spinCtrl_shiftSpeed_exponent.Bind( wx.EVT_SPINCTRL, self.m_spinCtrl_shiftSpeed_exponentOnUpdate )
		self.m_spinCtrl_shiftSpeed_exponent.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrl_shiftSpeed_exponentOnUpdate )
		self.m_slider_rotationSpeed.Bind( wx.EVT_SLIDER, self.m_slider_rotationSpeedOnSlider )
		self.m_spinCtrlDouble_rotationSpeed.Bind( wx.EVT_SPINCTRLDOUBLE, self.m_spinCtrlDouble_shiftSpeed_OnUpdate )
		self.m_spinCtrlDouble_rotationSpeed.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrlDouble_shiftSpeed_OnUpdate )
		self.m_button_saveScreeshot.Bind( wx.EVT_BUTTON, self.m_button_saveScreenshotOnButtonClick )
		self.m_textCtrl_screenshotPrefx.Bind( wx.EVT_TEXT, self.m_textCtrl_screenshotPrefixOnText )
		self.m_button_chooseScreenshotFolder.Bind( wx.EVT_BUTTON, self.m_button_chooseScreenshotFolderOnButtonClick )
		self.m_button_help.Bind( wx.EVT_BUTTON, self.m_button_helpOnButtonClick )
		self.m_button_nearestNeighbor.Bind( wx.EVT_BUTTON, self.m_button_nearestNeighborOnButtonClick )
		self.m_button_display.Bind( wx.EVT_BUTTON, self.m_button_displayOnButtonClick )
		self.m_button_newScene.Bind( wx.EVT_BUTTON, self.m_button_newSceneOnButtonClick )
		self.m_spinCtrlDouble_animationInterval.Bind( wx.EVT_SPINCTRLDOUBLE, self.m_spinCtrlDouble_animationIntervalOnUpdate )
		self.m_spinCtrlDouble_animationInterval.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrlDouble_animationIntervalOnUpdate )
		self.m_spinCtrlDouble_numberOfPoints.Bind( wx.EVT_SPINCTRLDOUBLE, self.m_spinCtrlDouble_numberOfPointsOnUpdate )
		self.m_spinCtrlDouble_numberOfPoints.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrlDouble_numberOfPointsOnUpdate )
		self.m_spinCtrlDouble_neighborhoodRadius_mantissa.Bind( wx.EVT_SPINCTRLDOUBLE, self.m_spinCtrlDouble_neighborhoodRadius_mantissaOnUpdate )
		self.m_spinCtrlDouble_neighborhoodRadius_mantissa.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrlDouble_neighborhoodRadius_mantissaOnUpdate )
		self.m_spinCtrl_neighborhoodRadius_exponent.Bind( wx.EVT_SPINCTRL, self.m_spinCtrl_neighborhoodRadius_exponentOnUpdate )
		self.m_spinCtrl_neighborhoodRadius_exponent.Bind( wx.EVT_TEXT_ENTER, self.m_spinCtrl_neighborhoodRadius_exponentOnUpdate )
		self.m_checkBox_shared_neighborhood.Bind( wx.EVT_CHECKBOX, self.m_checkBox_shared_neighborhoodOnCheckBox )

	def __del__( self ):
		pass


	# Virtual event handlers, override them in your derived class
	def m_spinCtrlDouble_visualSize_mantissaOnUpdate( self, event ):
		event.Skip()


	def m_spinCtrl_visualSize_exponentOnUpdate( self, event ):
		event.Skip()


	def m_spinCtrlDouble_shiftSpeed_mantissaOnUpdate( self, event ):
		event.Skip()


	def m_spinCtrl_shiftSpeed_exponentOnUpdate( self, event ):
		event.Skip()


	def m_slider_rotationSpeedOnSlider( self, event ):
		event.Skip()

	def m_spinCtrlDouble_shiftSpeed_OnUpdate( self, event ):
		event.Skip()


	def m_button_saveScreenshotOnButtonClick( self, event ):
		event.Skip()

	def m_textCtrl_screenshotPrefixOnText( self, event ):
		event.Skip()

	def m_button_chooseScreenshotFolderOnButtonClick( self, event ):
		event.Skip()

	def m_button_helpOnButtonClick( self, event ):
		event.Skip()

	def m_button_nearestNeighborOnButtonClick( self, event ):
		event.Skip()

	def m_button_displayOnButtonClick( self, event ):
		event.Skip()

	def m_button_newSceneOnButtonClick( self, event ):
		event.Skip()

	def m_spinCtrlDouble_animationIntervalOnUpdate( self, event ):
		event.Skip()


	def m_spinCtrlDouble_numberOfPointsOnUpdate( self, event ):
		event.Skip()


	def m_spinCtrlDouble_neighborhoodRadius_mantissaOnUpdate( self, event ):
		event.Skip()


	def m_spinCtrl_neighborhoodRadius_exponentOnUpdate( self, event ):
		event.Skip()


	def m_checkBox_shared_neighborhoodOnCheckBox( self, event ):
		event.Skip()


