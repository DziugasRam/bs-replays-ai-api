import math
from packaging.version import parse
import numpy as np
from scipy.special import comb
import time
import copy

# Works for both V2 and V3
# Easy = 1, Normal = 3, Hard = 5, Expert = 7, Expert+ = 9
# b = time, x and y = grid location from bottom left, a = angle offset, c = left or right respectively, d = cur direction
cut_direction_index = [90, 270, 180, 0, 135, 45, 225, 315, 270] #UP, DOWN, LEFT, RIGHT, TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT, DOT
right_handed_angle_strain_forehand = 247.5      # Most comfortable angle to aim for right hand (BOTTOM LEFT) 270 - 45 or 247.5
left_handed_angle_strain_forehand = 292.5       # 270 + 45 or 292.5
# 
# 
# 
def average(lst):   # Returns the averate of a list of integers
    if len(lst) > 0:
        return sum(lst) / len(lst)
    else:
        return 0
def bernstein_poly(i, n, t):    # For later
    """
     The Bernstein polynomial of n, i as a function of t
    """
    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i
def bezier_curve(points, nTimes=1000):   # For later
    """
       Given a set of control points, return the
       bezier curve defined by the control points.

       points should be a list of lists, or list of tuples
       such as [ [1,1], 
                 [2,3], 
                 [4,5], ..[Xn, Yn] ]
        nTimes is the number of time steps, defaults to 1000

        See http://processingjs.nihongoresources.com/bezierinfo/
    """

    nPoints = len(points)
    xPoints = np.array([p[0] for p in points])
    yPoints = np.array([p[1] for p in points])

    t = np.linspace(0.0, 1.0, nTimes)

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return list(xvals), list(yvals)
def V2_to_V3(V2mapData: dict):    # Convert V2 JSON to V3
    newMapData = {'colorNotes':[], 'bombNotes':[], 'obstacles':[]}  # I have to initialize this before hand or python gets grumpy
    for i in range(0, len(V2mapData['_notes'])):
        if V2mapData['_notes'][i]['_type'] in [0, 1]:   # In V2, Bombs and Notes were stored in the same _type key. The "if" just separates them
            newMapData['colorNotes'].append({'b': V2mapData['_notes'][i]['_time']})     # Append to make a new entry into the list to store the dictionary
            newMapData['colorNotes'][-1]['x'] = V2mapData['_notes'][i]['_lineIndex']
            newMapData['colorNotes'][-1]['y'] = V2mapData['_notes'][i]['_lineLayer']
            newMapData['colorNotes'][-1]['a'] = 0                                       # Angle offset didn't exist in V2. will always be 0
            newMapData['colorNotes'][-1]['c'] = V2mapData['_notes'][i]['_type']
            newMapData['colorNotes'][-1]['d'] = V2mapData['_notes'][i]['_cutDirection']
        elif V2mapData['_notes'][i]['_type'] == 3:      # Bombs
            newMapData['bombNotes'].append({'b': V2mapData['_notes'][i]['_time']})
            newMapData['bombNotes'][-1]['x'] = V2mapData['_notes'][i]['_lineIndex']
            newMapData['bombNotes'][-1]['y'] = V2mapData['_notes'][i]['_lineLayer']
    for i in range (0, len(V2mapData['_obstacles'])):
        newMapData['obstacles'].append({'b': V2mapData['_obstacles'][i]['_time']}) 
        newMapData['obstacles'][-1]['x'] = V2mapData['_obstacles'][i]['_lineIndex']
        if V2mapData['_obstacles'][i]['_type']:  # V2 wall type defines crouch or full walls
            newMapData['obstacles'][-1]['y'] = 2
            newMapData['obstacles'][-1]['h'] = 3
        else:
            newMapData['obstacles'][-1]['y'] = 0
            newMapData['obstacles'][-1]['h'] = 5
        newMapData['obstacles'][-1]['d'] = V2mapData['_obstacles'][i]['_duration']
        newMapData['obstacles'][-1]['w'] = V2mapData['_obstacles'][i]['_width']
    return newMapData
def splitMapData(mapData: dict, leftOrRight: int):    # False or 0 = Left, True or 1 = Right, 2 = Bombs
    if leftOrRight == 0:
        bloqList = [block for block in mapData['colorNotes'] if block['c'] == 0]  #Right handed blocks
    elif leftOrRight == 1:
        bloqList = [block for block in mapData['colorNotes'] if block['c'] == 1]  #Left handed blocks
    else:
        bloqList = [bomb for bomb in mapData['bombNotes']]
    return bloqList
def calculateBaseEntryExit(cBlockP, cBlockA):
    entry = [cBlockP[0] * 0.333333 - math.cos(math.radians(cBlockA)) * 0.166667 + 0.166667, cBlockP[1] * 0.333333 - math.sin(math.radians(cBlockA)) * 0.166667 + 0.16667]
    exit = [cBlockP[0] * 0.333333 + math.cos(math.radians(cBlockA)) * 0.166667 + 0.166667, cBlockP[1] * 0.333333 + math.sin(math.radians(cBlockA)) * 0.166667 + 0.16667]
    return entry, exit
def swingProcesser(mapSplitData: list):    # Returns a list of dictionaries for all swings returning swing angles and timestamps
    swingData = []
    for i in range(0, len(mapSplitData)):
        isSlider = False
        cBlockB = mapSplitData[i]['b']      # Current Block Position in Time in unit [Beats]        
        cBlockA = cut_direction_index[mapSplitData[i]['d']] + mapSplitData[i]['a']      # Current Block Angle in degrees
        cBlockP = [mapSplitData[i]['x'], mapSplitData[i]['y']]
        if i > 0:
            pBlockB = mapSplitData[i-1]['b']    # Pre-cache data for neater code
            pBlockA = swingData[-1]['angle'] # Previous Block Angle in degrees
            pBlockP = [mapSplitData[i-1]['x'], mapSplitData[i-1]['y']]
            if mapSplitData[i]['d'] == 8:   #Dot note? Just assume opposite angle. If it's a slider, the program will handle it
                pBlockA = swingData[-1]['angle']
                if cBlockB - pBlockB <= 0.03125:
                    cBlockA = pBlockA
                else:
                    if pBlockA >= 180:
                        cBlockA = pBlockA - 180
                    else:
                        cBlockA = pBlockA + 180
            # All Pre-caching Done
            if cBlockB - pBlockB >= 0.03125: # = 1/32 Just a check if notes are unreasonable close, just assume they're apart of the same swing
                if cBlockB - pBlockB > 0.125: # = 1/8 The upper bound of normal slider precision commonly used
                    if cBlockB - pBlockB > 0.5:    # = 1/2 About the limit of whats reasonable to expect from a slider
                        swingData.append({'time': cBlockB, 'angle': cBlockA})
                        swingData[-1]['entryPos'], swingData[-1]['exitPos'] = calculateBaseEntryExit(cBlockP, cBlockA)
                    else: # 1/2 Check (just to weed out obvious non-sliders) More complicated methods need to be used
                        if abs(cBlockA - pBlockA) < 112.5:  # 90 + 22.5 JUST IN CASE. not the full 90 + 45 since that would be one hell of a slider or dot note
                            testAnglefromPosition = math.degrees(math.atan2(pBlockP[1]-cBlockP[1], pBlockP[0]-cBlockP[0])) % 360 # Replaces angle swing from block angle to slider angle
                            averageAngleOfBlocks = (cBlockA + pBlockA) / 2
                            if abs(testAnglefromPosition - averageAngleOfBlocks) <= 56.25:  # = 112.5 / 2 = 56.25
                                sliderTime = cBlockB - pBlockB
                                isSlider = True
                            else:
                                swingData.append({'time': cBlockB, 'angle': cBlockA})       # Below calculates the entry and exit positions for each swing
                                swingData[-1]['entryPos'], swingData[-1]['exitPos'] = calculateBaseEntryExit(cBlockP, cBlockA)
                        else:
                            swingData.append({'time': cBlockB, 'angle': cBlockA})
                            swingData[-1]['entryPos'], swingData[-1]['exitPos'] = calculateBaseEntryExit(cBlockP, cBlockA)
                else: # 1/8 Check
                    if mapSplitData[i]['d'] == 8 or abs(cBlockA - pBlockA) < 90: # 90 degree check since 90 degrees is what most would consider the maximum angle for a slider or dot note
                        sliderTime = 0.125
                        isSlider = True
                    else:
                        swingData.append({'time': cBlockB, 'angle': cBlockA})
                        swingData[-1]['entryPos'], swingData[-1]['exitPos'] = calculateBaseEntryExit(cBlockP, cBlockA)
            else:   # 1/32 Check
                sliderTime = 0.03125
                isSlider = True
            if isSlider:
                for f in range(1, len(mapSplitData)):   # We clearly know the last block is a slider with the current block under test. Skip to the one before the last block. Should realistically never search more than 5 blocks deep
                    blockIndex = i - f              # Index of the previous block to start comparisons with
                    if blockIndex < 0:
                        break      # We Reached the beginning of the map
                    if (mapSplitData[blockIndex]['b'] - mapSplitData[blockIndex - 1]['b'] > 2 * sliderTime):       # use 2x slider time to account for any "irregularities" / margin of error. We are only comparing pairs of blocks
                        pBlockB = mapSplitData[blockIndex]['b']                                             # Essentially finds then caches first block in the slider group
                        pBlockA = mapSplitData[blockIndex]['a']
                        pBlockP = [mapSplitData[blockIndex]['x'], mapSplitData[blockIndex]['y']]
                        break
                
                cBlockA = math.degrees(math.atan2(pBlockP[1]-cBlockP[1], pBlockP[0]-cBlockP[0])) % 360 # Replaces angle swing from block angle to slider angle
                guideAngle = 1150           # A random test value to check later
                for f in range(1, len(mapSplitData)):       # Checker that will try to find a guiding block (arrow block) for the slider angle prediction.
                    blockIndex = i - f
                    if mapSplitData[blockIndex]['b'] < pBlockB:     # Limits check to a little after the first slider block in the group
                        break
                    if mapSplitData[blockIndex]['d'] != 8:          # Breaks out of loop when it finds an arrow block
                        guideAngle = cut_direction_index[mapSplitData[blockIndex]['d']]
                        break
                if guideAngle != 1150:      # A test to see if guideAngle was actually changed
                    if abs(cBlockA - guideAngle) > 90:       # If this is true, the predicted angle is wrong, likely by 180 degrees wrong
                        if cBlockA >= 180:
                            cBlockA -= 180               # Apply Fix
                        else:
                            cBlockA += 180
                swingData[-1]['angle'] = cBlockA
                
                xtest = (swingData[-1]['entryPos'][0] - (cBlockP[0] * 0.333333 - math.cos(math.radians(cBlockA)) * 0.166667 + 0.166667)) * math.cos(math.radians(cBlockA))
                ytest = (swingData[-1]['entryPos'][1] - (cBlockP[1] * 0.333333 - math.sin(math.radians(cBlockA)) * 0.166667 + 0.166667)) * math.sin(math.radians(cBlockA))
                if xtest <= 0.001 and ytest >= 0.001:       # For sliders, one of the entry/exit positions is still correct, this figures out which one then replaces the other
                    swingData[-1]['entryPos'] = [cBlockP[0] * 0.333333 - math.cos(math.radians(cBlockA)) * 0.166667 + 0.166667, cBlockP[1] * 0.333333 - math.sin(math.radians(cBlockA)) * 0.166667 + 0.16667]
                else:
                    swingData[-1]['exitPos'] = [cBlockP[0] * 0.333333 + math.cos(math.radians(cBlockA)) * 0.166667 + 0.166667, cBlockP[1] * 0.333333 + math.sin(math.radians(cBlockA)) * 0.166667 + 0.16667]   
        else:
            swingData.append({'time': cBlockB, 'angle': cBlockA})    # First Note Exception. will never be a slider or need to undergo any test
            swingData[-1]['entryPos'], swingData[-1]['exitPos'] = calculateBaseEntryExit(cBlockP, cBlockA)
    return swingData
def swingAngleStrainCalc(swingData: list, leftOrRight): # False or 0 = Left, True or 1 = Right
    strainAmount = 0
    #TODO calculate strain from angle based on left or right hand
    for i in range(0, len(swingData)):
        if swingData[i]['forehand']:     #The Formula firse calculates by first normalizing the angle difference (/180) then using
            if leftOrRight:
                strainAmount += 2 * (((180 - abs(abs(right_handed_angle_strain_forehand - swingData[i]['angle']) - 180)) / 180)**2)          # Right Handed Forehand
            else:
                strainAmount += 2 * (((180 - abs(abs(left_handed_angle_strain_forehand - swingData[i]['angle']) - 180)) / 180)**2)           # Left Handed Forehand
        else:
            if leftOrRight:
                strainAmount += 2 * (((180 - abs(abs(right_handed_angle_strain_forehand - 180 - swingData[i]['angle']) - 180))/180)**2)           # Right Handed Backhand
            else:
                strainAmount += 2 * (((180 - abs(abs(left_handed_angle_strain_forehand - 180 - swingData[i]['angle']) - 180))/180)**2)           # Left Handed Backhand
    return strainAmount * 2
def bezierAngleStrainCalc(angleData: list, forehand, leftOrRight):
    strainAmount = 0
    for i in range(0, len(angleData)):
        if forehand:
            if leftOrRight:
                strainAmount += 2 * (((180 - abs(abs(right_handed_angle_strain_forehand - angleData[i]) - 180)) / 180)**2)          # Right Handed Forehand
            else:
                strainAmount += 2 * (((180 - abs(abs(left_handed_angle_strain_forehand - angleData[i]) - 180)) / 180)**2)           # Left Handed Forehand
        else:
            if leftOrRight:
                strainAmount += 2 * (((180 - abs(abs(right_handed_angle_strain_forehand - 180 - angleData[i]) - 180))/180)**2)           # Right Handed Backhand
            else:
                strainAmount += 2 * (((180 - abs(abs(left_handed_angle_strain_forehand - 180 - angleData[i]) - 180))/180)**2)           # Left Handed Backhand
    return strainAmount
def patternSplitter(swingData: list):    # Does swing speed analysis to split the long list of dictionaries into smaller lists of patterns containing lists of dictionaries
    for i in range(0, len(swingData)):   # Swing Frequency Analyzer
        if i > 0 and i+1 < len(swingData):    # Checks done so we don't try to access data that doesn't exist
            SF = 2/(swingData[i+1]['time'] - swingData[i-1]['time'])    # Swing Frequency
        else:
            SF = 0
        swingData[i]['frequency'] = SF
    patternFound = False
    SFList = [freq['frequency'] for freq in swingData]
    SFmargin = average(SFList) / 32
    patternList = []            # Pattern List
    tempPlist = []              # Temp Pattern List
    for i in range(0, len(swingData)):
        if i > 0:
            if (1 / (swingData[i]['time'] - swingData[i-1]['time'])) - swingData[i]['frequency'] <= SFmargin:    # Tries to find Patterns within margin
                if not patternFound:    # Found a pattern and it's the first one?
                    patternFound = True
                    del tempPlist[-1]
                    if len(tempPlist) > 0:  # We only want to store lists with stuff
                        patternList.append(tempPlist)
                    tempPlist = [swingData[i-1]]    #Store the 1st block of the pattern
                tempPlist.append(swingData[i])  # Store the block we're working on
            else:
                if len(tempPlist) > 0 and patternFound:
                    tempPlist.append(swingData[i])
                    patternList.append(tempPlist)
                    tempPlist = []
                else:
                    patternFound = False
                    tempPlist.append(swingData[i])
        else:
            tempPlist.append(swingData[0])
    return patternList
def parityPredictor(patternData: list, bombData: list, leftOrRight):    # Parses through a List of Lists of Dictionaries to calculate the most likely parity for each pattern
    newPatternData = []
    for p in range(0, len(patternData)):
        testData1 = patternData[p]
        testData2 = copy.deepcopy(patternData[p])
        for i in range(0, len(testData1)):  # Build Forehand TestData Build
            if i > 0:
                if abs(testData1[i]['angle'] - testData1[i-1]['angle']) > 45:     # If angles are too similar, assume reset since a write roll of that degree is crazy
                    testData1[i]['forehand'] = not testData1[i-1]['forehand']
                else:
                    testData1[i]['forehand'] = testData1[i-1]['forehand']
            else:
                testData1[0]['forehand'] = True
        for i in range(0, len(testData2)):  # Build Banckhand TestData
            if i > 0:
                if abs(testData2[i]['angle'] - testData2[i-1]['angle']) > 45:     # Again, if angles are too similar, assume reset since a write roll of that degree is crazy
                    testData2[i]['forehand'] = not testData2[i-1]['forehand']
                else:
                    testData2[i]['forehand'] = testData2[i-1]['forehand']
            else:
                testData2[0]['forehand'] = False
        forehandTest = swingAngleStrainCalc(testData1, leftOrRight)    # Test Data
        backhandTest = swingAngleStrainCalc(testData2, leftOrRight)    # 
        if forehandTest <= backhandTest:    #Prefer forehand starts over backhand if equal
            newPatternData += testData1      # Forehand gave a lower stress value, therefore is the best option in terms of hand placement for the pattern
        elif forehandTest > backhandTest:
            newPatternData += testData2
    for i in range(0, len(newPatternData)):
        newPatternData[i]['angleStrain'] = swingAngleStrainCalc([newPatternData[i]], leftOrRight)  # Assigns individual strain values to each swing. Done like this in square brackets because the function expects a list.
        if i > 0:
            if newPatternData[i]['forehand'] == newPatternData[i-1]['forehand']:
                newPatternData[i]['reset'] = True
            else:
                newPatternData[i]['reset'] = False
    return newPatternData
def staminaCalc(swingData: list):
    staminaList: list = []
    #TODO calculate strain from stamina drain


    return staminaList
def swingCurveCalc(swingData: list, leftOrRight, isuser=True):
    if len(swingData) == 0:
        return swingData
    swingData[0]['pathStrain'] = 0  # First Note cannot really have any path strain
    testData = []
    for i in range(1, len(swingData)):
        point0 = swingData[i-1]['exitPos']      # Curve Beginning
        point1x = point0[0] + 0.5 * math.cos(math.radians(swingData[i-1]['angle']))
        point1y = point0[1] + 0.5 * math.sin(math.radians(swingData[i-1]['angle']))
        point1 = [point1x, point1y] #Curve Control Point
        point3 = swingData[i]['entryPos']       # Curve Ending
        point2x = point3[0] - 0.5 * math.cos(math.radians(swingData[i]['angle']))
        point2y = point3[1] - 0.5 * math.sin(math.radians(swingData[i]['angle']))
        point2 = [point2x, point2y]     #Curve Control Point
        points = [point0, point1, point2, point3]
        xvals, yvals = bezier_curve(points, nTimes=50)
        xvals.reverse()
        yvals.reverse()
        speedList = []
        angleList = []
        for f in range(1, min(len(xvals), len(yvals))): 
            speedList.append(math.sqrt((yvals[f] - yvals[f-1])**2 + (xvals[f] - xvals[f-1])**2))
            angleList.append(math.degrees(math.atan2(yvals[f] - yvals[f-1], xvals[f] - xvals[f-1])) % 360)
        if swingData[i]['reset']:       # If the pattern is a reset, look less far back
            lookback = 0.80                     # 0.5 angle strain = 0.35 or 65% lookback, 0.1 angle strain = 0.5 or 50% lookback
        else:
            lookback = 0.333333
        curveComplexity = len(speedList) * average(speedList[int(len(speedList) * lookback):]) / 20
        pathAngleStrain = bezierAngleStrainCalc(angleList[int(len(angleList) * lookback):], swingData[i]['forehand'], leftOrRight) / len(angleList) * 2

        testData.append({'curveComplexityStrain': curveComplexity, 'pathAngleStrain': pathAngleStrain})
        swingData[i]['pathStrain'] = curveComplexity + pathAngleStrain
    if leftOrRight:
        hand = 'Right Handed'
    else:
        hand = 'Left Handed'
    if isuser:
        print(f"Average {hand} hitAngleStrain {average([Stra['angleStrain'] for Stra in swingData])}")
        print(f"Average {hand} curveComplexityStrain {average([Stra['curveComplexityStrain'] for Stra in testData])}")
        print(f"Average {hand} pathAngleStrain {average([Stra['pathAngleStrain'] for Stra in testData])}")


    return swingData
def combineAndSortList(array1, array2, key):
    combinedArray = array1 + array2
    combinedArray = sorted(combinedArray, key=lambda x: x[f'{key}'])  # once combined, sort by time
    return combinedArray
    songPath = findSongPath(mapID, isuser)
    diffList = findStandardDiffs(songPath)
    if diffNum in diffList:     # Check if the song is listed in the Info.dat file, otherwise exits programs
        diffPath = diffNum_to_diffPath(songPath, diffNum)
        mapData = load_json_as_dict(diffPath)
        return mapData
    else:
        print(f"Map {mapID} Diff {diffNum} doesn't exist locally. Are you sure you have the updated version?")
        print("Enter to Exit")
        input()
        exit()
def techOperations(mapData, isuser=True):
    LeftMapData = splitMapData(mapData, 0)
    RightMapData = splitMapData(mapData, 1)
    bombData = splitMapData(mapData, 2)
    
    LeftSwingData = swingProcesser(LeftMapData)
    RightSwingData = swingProcesser(RightMapData)
    
    LeftPatternData = patternSplitter(LeftSwingData)
    RightPatternData = patternSplitter(RightSwingData)
    
    LeftSwingData = parityPredictor(LeftPatternData, bombData, False)
    RightSwingData = parityPredictor(RightPatternData, bombData, True)
    
    LeftSwingData = swingCurveCalc(LeftSwingData, False, isuser)
    RightSwingData = swingCurveCalc(RightSwingData, True, isuser)
    
    SwingData = combineAndSortList(LeftSwingData, RightSwingData, 'time')
    StrainList = [strain['angleStrain'] + strain['pathStrain'] for strain in SwingData]
    StrainList.sort()
    tech = average(StrainList[int(len(StrainList) * 0.25):])
    
    if isuser:
        print(f"Calculacted Tech = {tech}")        # Put Breakpoint here if you want to see
    return tech

def techCalculation(mapData, isuser=True):
    t0 = time.time()
    try:
        mapVersion = parse(mapData['version'])
    except KeyError:
        try:
            mapVersion = parse(mapData['_version'])
        except KeyError:
            try:
                mapData['_notes']
                mapVersion = parse('2.0.0')
            except KeyError:
                mapData['colorNotes']
                mapVersion = parse('3.0.0')

    if mapVersion < parse('3.0.0'):     # Try to figure out if the map is the V2 or V3 format
        maptype = 2
        mapData = V2_to_V3(mapData)     # Convert to V3
    else:
        maptype = 3
    tech = techOperations(mapData, isuser)
    t1 = time.time()
    print(f'Execution Time = {t1-t0}')
    return tech
