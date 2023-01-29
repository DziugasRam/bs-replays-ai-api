import math
from packaging.version import parse
import numpy as np
from scipy.special import comb
import time
import copy
from collections import deque

# Works for both V2 and V3
# Easy = 1, Normal = 3, Hard = 5, Expert = 7, Expert+ = 9
# b = time, x and y = grid location from bottom left, a = angle offset, c = left or right respectively, d = cur direction
cut_direction_index = [90, 270, 180, 0, 135, 45, 225, 315, 270] #UP, DOWN, LEFT, RIGHT, TOPLEFT, TOPRIGHT, BOTTOMLEFT, BOTTOMRIGHT, DOT
right_handed_angle_strain_forehand = 247.5      # Most comfortable angle to aim for right hand (BOTTOM LEFT) 270 - 45 or 247.5
left_handed_angle_strain_forehand = 292.5       # 270 + 45 or 292.5
# 
# 

def average(lst, setLen=0):   # Returns the averate of a list of integers
    if len(lst) > 0:
        if setLen == 0:
            return sum(lst) / len(lst)
        else:
            return sum(lst) / setLen
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
def mapPrep(mapData):
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
                try:
                    mapData['colorNotes']
                    mapVersion = parse('3.0.0')
                except KeyError:
                    print("Unknown Map Type. Exiting")
                    exit()
    if mapVersion < parse('3.0.0'):     # Try to figure out if the map is the V2 or V3 format
        newMapData = V2_to_V3(mapData)     # Convert to V3
    else:
        newMapData = mapData
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
    mapSplitData = sorted(mapSplitData, key=lambda d: d['b'])
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
                        if mapSplitData[i]['d'] != 8:
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
                        else:       # Dot note thats also close? Swing detection gets a little more tricky
                            if i < len(mapSplitData) - 1:   # Checks if the next memory access is legal
                                if (cBlockB - pBlockB) - (mapSplitData[i+1]['b'] - cBlockB) < 0.125:    # Checks which swing the dot note is associated with.
                                    sliderTime = cBlockB - pBlockB
                                    isSlider = True
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
                    if (mapSplitData[blockIndex]['b'] - mapSplitData[blockIndex - 1]['b'] > 1.5 * sliderTime):       # use 2x slider time to account for any "irregularities" / margin of error. We are only comparing pairs of blocks
                        pBlockB = mapSplitData[blockIndex]['b']                                             # Essentially finds then caches first block in the slider group
                        pBlockA = cut_direction_index[mapSplitData[blockIndex]['d']] + mapSplitData[blockIndex]['a']
                        pBlockP = [mapSplitData[blockIndex]['x'], mapSplitData[blockIndex]['y']]
                        break
                    if(mapSplitData[blockIndex]['d'] != 8):     
                        break
                
                cBlockA = math.degrees(math.atan2(pBlockP[1]-cBlockP[1], pBlockP[0]-cBlockP[0])) % 360 # Replaces angle swing from block angle to slider angle
                if len(swingData) > 1:
                    guideAngle = (swingData[-2]['angle'] - 180) % 360           # Use the opposite swing angle as a base starting point
                else:
                    guideAngle = 270        # First swing? use downward swing as a base starting guide
                for f in range(1, len(mapSplitData)):       # Checker that will try to find a better guiding block (arrow block) for the slider angle prediction.
                    blockIndex = i - f
                    if mapSplitData[blockIndex]['b'] < pBlockB:     # Limits check to a little after the first slider block in the group
                        break
                    if mapSplitData[blockIndex]['d'] != 8:          # Breaks out of loop when it finds an arrow block
                        guideAngle = cut_direction_index[mapSplitData[blockIndex]['d']]     # If you found an arrow, use it's angle
                        break
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
                if abs(testData1[i]['angle'] - testData1[i-1]['angle']) >= 67.5:     # If angles are too similar, assume reset since a write roll of that degree is crazy
                    testData1[i]['forehand'] = not testData1[i-1]['forehand']
                else:
                    testData1[i]['forehand'] = testData1[i-1]['forehand']
            else:
                testData1[0]['forehand'] = True
        for i in range(0, len(testData2)):  # Build Banckhand TestData
            if i > 0:
                if abs(testData2[i]['angle'] - testData2[i-1]['angle']) >= 67.5:     # = 45 + 22.5
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
        else:
            newPatternData[i]['reset'] = False
    return newPatternData
def staminaCalc(data: list):
    swingDiffList = [temp['swingDiff'] for temp in data]
    swingDiffList.sort(reverse=True)
    averageDiff = average(swingDiffList[:int(len(swingDiffList) * 0.5)])
    burstDiff = average(swingDiffList[:min(round(len(swingDiffList) / 8), 1)])
    if burstDiff == 0:
        return 0
    staminaRatio = averageDiff / burstDiff
    return 1 / (10 + 4**(-64 * (staminaRatio - 0.875))) + 0.9 + staminaRatio / 20
def swingCurveCalc(swingData: list, leftOrRight, isuser=True):
    if len(swingData) == 0:
        returnDict = {'hitAngleStrain': 0, 'positionComplexity': 0, 'curveComplexityStrain': 0, 'pathAngleStrain': 0}
        return swingData, returnDict
    swingData[0]['pathStrain'] = 0  # First Note cannot really have any path strain
    testData = []
    for i in range(1, len(swingData)):
        point0 = swingData[i-1]['exitPos']      # Curve Beginning
        point1x = point0[0] + 1 * math.cos(math.radians(swingData[i-1]['angle']))
        point1y = point0[1] + 1 * math.sin(math.radians(swingData[i-1]['angle']))
        point1 = [point1x, point1y] #Curve Control Point
        point3 = swingData[i]['entryPos']       # Curve Ending
        point2x = point3[0] - 1 * math.cos(math.radians(swingData[i]['angle']))
        point2y = point3[1] - 1 * math.sin(math.radians(swingData[i]['angle']))
        point2 = [point2x, point2y]     #Curve Control Point
        points = [point0, point1, point2, point3]
        xvals, yvals = bezier_curve(points, nTimes=25)      #nTimes = the resultion of the bezier curve. Higher = more accurate but slower
        xvals.reverse()
        yvals.reverse()
        positionComplexity = 0
        angleChangeList = []
        angleList = []
        distance = 0
        for f in range(1, min(len(xvals), len(yvals))):
            angleList.append(math.degrees(math.atan2(yvals[f] - yvals[f-1], xvals[f] - xvals[f-1])) % 360)
            distance += math.sqrt((yvals[f] - yvals[f-1])**2 + (xvals[f] - xvals[f-1])**2)
            if f > 1:
                angleChangeList.append(180 - abs(abs(angleList[-1] - angleList[-2]) - 180))   # Wacky formula to handle 5 - 355 situations
        distance -= 0.75

        if i > 1:       # Will miss the very first reset if it exists but a sacrafice for speed
            simHandCurPos = swingData[i]['entryPos']
            if(swingData[i]['forehand'] == swingData[i-2]['forehand']):     #Start 2 swings back since it's the most likely
                simHandPrePos = swingData[i-2]['entryPos']
            elif(swingData[i]['forehand'] == swingData[i-1]['forehand']):
                simHandPrePos = swingData[i-1]['entryPos']
            else:
                simHandPrePos = simHandCurPos
            positionDiff = math.sqrt((simHandCurPos[1] - simHandPrePos[1])**2 + (simHandCurPos[0] - simHandPrePos[0])**2)
            positionComplexity = positionDiff**2

        lengthOfList = len(angleChangeList) * (1 - 0.4)             # 0.2 + (1 - 0.8) = 0.4

        
        if swingData[i]['reset']:       # If the pattern is a reset, look less far back
            pathLookback = 0.9
            first = 0.5
            last = 1
        else:
            pathLookback = 0.5
            first = 0.2
            last = 0.8
        pathLookbackIndex = int(len(angleList) * pathLookback)
        firstIndex = int(len(angleChangeList)* first) - 1
        lastIndex = int(len(angleChangeList)* last) - 1

        curveComplexity = abs((lengthOfList * average(angleChangeList[firstIndex:lastIndex]) - 180) / 180)   # The more the angle difference changes from 180, the more complex the path, /180 to normalize between 0 - 1
        pathAngleStrain = bezierAngleStrainCalc(angleList[pathLookbackIndex:], swingData[i]['forehand'], leftOrRight) / len(angleList) * 2

        # print(f"positionComplexity {positionComplexity}")
        # print(f"curveComplexity {curveComplexity}")
        # print(f"pathAngleStrain {pathAngleStrain}")
        # from matplotlib import pyplot as plt        #   Test
        # fig, ax = plt.subplots(figsize = (8, 5))
        # ax.plot(xvals, yvals, label='curve path')
        # xpoints = [p[0] for p in points]
        # ypoints = [p[1] for p in points]
        # ax.plot(xvals, yvals, label='curve path')
        # ax.plot(xpoints, ypoints, "ro", label='Control Points')
        # ax.plot(xvals[int(len(xvals) * pathLookback)], yvals[int(len(yvals) * pathLookback)], "bo", label='pathAngleStrain Start Point')
        # ax.plot([xvals[int(len(xvals) * first) - 1], xvals[int(len(xvals) * last) - 1]], [yvals[int(len(yvals) * first) - 1], yvals[int(len(yvals) * last) - 1]], 'go', label='curveComplexity Scope')
        # ax.set_xticks(np.linspace(0,1.333333333,5))
        # ax.set_yticks(np.linspace(0,1,4))
        # #plt.xlim(0,1.3333333)
        # #plt.ylim(0,1)
        # plt.legend()
        # plt.show()

        testData.append({'curveComplexityStrain': curveComplexity, 'pathAngleStrain': pathAngleStrain, 'positionComplexity': positionComplexity})
        swingData[i]['positionComplexity'] = positionComplexity
        swingData[i]['preDistance'] = distance
        swingData[i]['curveComplexity'] = curveComplexity
        swingData[i]['pathAngleStrain'] = pathAngleStrain
        swingData[i]['pathStrain'] = curveComplexity + pathAngleStrain + positionComplexity
    avehitAngleStrain = average([Stra['angleStrain'] for Stra in swingData])
    avepositionComplexity = average([Stra['positionComplexity'] for Stra in testData])
    avecurveComplexityStrain = average([Stra['curveComplexityStrain'] for Stra in testData])
    avepathAngleStrain = average([Stra['pathAngleStrain'] for Stra in testData])
    returnDict = {'hitAngleStrain': avehitAngleStrain, 'positionComplexity': avepositionComplexity, 'curveComplexityStrain': avecurveComplexityStrain, 'pathAngleStrain': avepathAngleStrain}
    if leftOrRight:
        hand = 'Right Handed'
    else:
        hand = 'Left Handed'
    if isuser:
        print(f"Average {hand} hitAngleStrain {avehitAngleStrain}")
        print(f"Average {hand} positionComplexity {avepositionComplexity}")
        print(f"Average {hand} curveComplexityStrain {avecurveComplexityStrain}")
        print(f"Average {hand} pathAngleStrain {avepathAngleStrain}")
    return swingData, returnDict
def diffToPass(swingData, bpm, hand, isuser=True):
    bps = bpm / 60
    # SSSpeed = 0         #Sum of Swing Speed
    # qSS = deque()       #List of swing speed
    # SSStress = 0             #Sum of swing stress
    # qST = deque()       #List of swing stress
    qDIFF = deque()
    WINDOW = 50       #Adjusts the smoothing window (how many swings get smoothed) (roughly 8 notes to fail)
    difficultyIndex = []
    data = []
    # for i in range(1, len(swingData)):      # Scan all swings, starting from 2nd swing
        # # xPathDist = swingData[i]['exitPos'][0] - swingData[i-1]['exitPos'][0]
        # # yPathDist = swingData[i]['exitPos'][1] - swingData[i-1]['exitPos'][1]
        # # data.append({'preDistance': math.sqrt((xPathDist**2) + (yPathDist**2))})
        # # distanceDiff = data[-1]['preDistance'] / (data[-1]['preDistance'] + 3) + 1
        # distanceDiff = swingData[i]['preDistance'] / (swingData[i]['preDistance'] + 3) + 1
        # if i > smoothing:       # Start removing old swings based on smoothing amount
        #     SSSpeed -= qSS.popleft()
        #     SSStress -= qST.popleft()
        # qSS.append(swingData[i]['frequency'] * distanceDiff * bps)
        # SSSpeed += qSS[-1]
        # # data[-1]['swingSpeedAve'] = SSSpeed / smoothing
        # data.append({'swingSpeedAve': SSSpeed / smoothing})

        # xHitDist = swingData[i]['entryPos'][0] - swingData[i]['exitPos'][0]
        # yHitDist = swingData[i]['entryPos'][1] - swingData[i]['exitPos'][1]
        # data[-1]['hitDistance'] = math.sqrt((xHitDist**2) + (yHitDist**2))
        # data[-1]['hitDiff'] =  data[-1]['hitDistance'] / (data[-1]['hitDistance'] + 2) + 1

        # qST.append((swingData[i]['angleStrain'] + swingData[i]['pathStrain']) * data[-1]['hitDiff'])
        # SSStress += qST[-1]
        # data[-1]['stressAve'] = SSStress / smoothing
        
        # difficulty = data[-1]['swingSpeedAve'] * (-1.4**(-data[-1]['swingSpeedAve']) + 1) * (data[-1]['stressAve'] / (data[-1]['stressAve'] + 2) + 1) * 0.80
        # #difficulty = data[-1]['swingSpeedAve'] + data[-1]['stressAve']
        # data[-1]['difficulty'] = difficulty
        # difficultyIndex.append(difficulty)

    # if isuser:
    #     peakSS = [temp['swingSpeedAve'] for temp in data]
    #     peakSS.sort(reverse=True)
    #     print(f"peak {hand} hand speed {average(peakSS[:int(len(peakSS) / 16)])}")
    #     print(f"average {hand} hand stress {average([temp['stressAve'] for temp in data])}")

    # difficultyIndex.sort(reverse=True)      #Sort list by most difficult
    # return average(difficultyIndex[:int(len(difficultyIndex) * 0.3)])          # Use the top 8 swings averaged as the return
    if len(swingData) == 0:
        return 0
    swingData[0]['swingDiff'] = 0
    for i in range(1, len(swingData)):
        distanceDiff = swingData[i]['preDistance'] / (swingData[i]['preDistance'] + 3) + 1
        data.append({'swingSpeed': swingData[i]['frequency'] * distanceDiff * bps})
        xHitDist = swingData[i]['entryPos'][0] - swingData[i]['exitPos'][0]
        yHitDist = swingData[i]['entryPos'][1] - swingData[i]['exitPos'][1]
        data[-1]['hitDistance'] = math.sqrt((xHitDist**2) + (yHitDist**2))
        data[-1]['hitDiff'] =  data[-1]['hitDistance'] / (data[-1]['hitDistance'] + 2) + 1
        data[-1]['stress'] = (swingData[i]['angleStrain'] + swingData[i]['pathStrain']) * data[-1]['hitDiff']
        swingData[i]['swingDiff'] = data[-1]['swingSpeed'] * (-1.4**(-data[-1]['swingSpeed']) + 1) * (data[-1]['stress'] / (data[-1]['stress'] + 2) + 1)

        if i > WINDOW:
            qDIFF.popleft()
        qDIFF.append(swingData[i]['swingDiff'])
        tempList = sorted(qDIFF, reverse=True)
        windowDiff = average(tempList[:int(len(tempList) * 25 / WINDOW)], 25) * 0.80        # Top 15 notes out of the window
        difficultyIndex.append(windowDiff)
    
    if isuser:
        peakSS = [temp['swingSpeed'] for temp in data]
        peakSS.sort(reverse=True)
        print(f"peak {hand} hand speed {average(peakSS[:int(len(peakSS) / 16)])}")
        print(f"average {hand} hand stress {average([temp['stress'] for temp in data])}")

    

    if len(difficultyIndex) > 0:
        return max(difficultyIndex) 
    else:
        return 0

def combineAndSortList(array1, array2, key):
    combinedArray = array1 + array2
    combinedArray = sorted(combinedArray, key=lambda x: x[f'{key}'])  # once combined, sort by time
    return combinedArray

def techOperations(mapData, bpm, isuser=True, verbose=True):

    LeftMapData = splitMapData(mapData, 0)
    RightMapData = splitMapData(mapData, 1)
    bombData = splitMapData(mapData, 2)
    
    LeftSwingData = swingProcesser(LeftMapData)
    RightSwingData = swingProcesser(RightMapData)
    
    LeftPatternData = patternSplitter(LeftSwingData)
    RightPatternData = patternSplitter(RightSwingData)
    
    LeftSwingData = parityPredictor(LeftPatternData, bombData, False)
    RightSwingData = parityPredictor(RightPatternData, bombData, True)
    
    LeftSwingData, leftVerbose = swingCurveCalc(LeftSwingData, False, isuser)
    RightSwingData, rightVerbose = swingCurveCalc(RightSwingData, True, isuser)
    
    SwingData = combineAndSortList(LeftSwingData, RightSwingData, 'time')
    StrainList = [strain['angleStrain'] + strain['pathStrain'] for strain in SwingData]
    StrainList.sort()
    tech = average(StrainList[int(len(StrainList) * 0.25):])
    passNum = max(diffToPass(LeftSwingData, bpm, 'left', isuser), diffToPass(RightSwingData, bpm, 'right', isuser))

    staminaFactor = average([staminaCalc(LeftSwingData), staminaCalc(RightSwingData)])
    balanced_pass = passNum * staminaFactor

    balanced_tech = tech * (-1.4**(-passNum) + 1)

    if verbose:
        returnDict = {'left': leftVerbose, 'right': rightVerbose, 'tech': tech, 'passing_difficulty': passNum, 'balanced_tech': balanced_tech, 'balanced_pass_diff': balanced_pass}
    else:
        returnDict = {'balanced_tech': balanced_tech, 'balanced_pass_diff': balanced_pass}
    if isuser:
        print(f"Calculacted Tech = {tech}")        # Put Breakpoint here if you want to see
        print(f"Calculacted stamina factor = {staminaFactor}")
        print(f"Calculated pass diff = {passNum}")
        print(f"Calculated balanced tech = {balanced_tech}")
        print(f"Calculated balanced pass diff = {balanced_pass}")
    return returnDict

def mapCalculation(mapData, bpm, isuser=True, verbose=True):
    t0 = time.time()
    newMapData = mapPrep(mapData)
    data = techOperations(newMapData, bpm, isuser, verbose)
    t1 = time.time()
    if isuser:
        print(f'Execution Time = {t1-t0}')
    return data