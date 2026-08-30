import QtQuick 6.5
import QtQuick3D 6.5
import QtQml.Models


View3D {
    id: worldView
    objectName: "worldView"

    property bool showAxes: true

    property real moveStepMm: 60
    property real yawStepDeg: 2.5
    property real pitchStepDeg: 2.5
    property real zoomStepGl: 2
    property real elevateStepMm: 20
    property real minZoomGl: 2
    property real maxZoomGl: 100
    property real minPitchDeg: -60
    property real maxPitchDeg: 60

    property real sceneYawDeg: 90
    property vector3d scenePosition: Qt.vector3d(0, 0, 0)
    property real cameraPitchDeg: -30
    property real cameraDistanceGl: 10
    property vector3d cameraPosition: Qt.vector3d(0, 0, 0)

    onScenePositionChanged: updateCamera()
    onSceneYawDegChanged: updateCamera()
    onCameraPitchDegChanged: updateCamera()
    onCameraDistanceGlChanged: updateCamera()

    readonly property real worldScale: Number(WSCALE) || 0.02

    function worldToSceneVector(xMm, yMm, zMm) {
        const s = worldScale
        return Qt.vector3d(xMm * s, zMm * s, -yMm * s)
    }

    function radiansToDegrees(angle) {
        return angle * 180 / Math.PI
    }

    function degreesToRadians(angle) {
        return angle * Math.PI / 180
    }

    function clamp(value, minValue, maxValue) {
        return Math.max(minValue, Math.min(maxValue, value))
    }

    function dominoPipColor(count) {
        switch (Number(count)) {
        case 1:
            return "#5a0034"
        case 2:
            return "#2f8f24"
        case 3:
            return "#5b197f"
        case 4:
            return "#009fc4"
        case 5:
            return "#2f8f24"
        case 6:
            return "#bd7b16"
        default:
            return "#202020"
        }
    }

    function dominoPipEmissive(count) {
        switch (Number(count)) {
        case 1:
            return Qt.vector3d(0.16, 0.00, 0.08)
        case 2:
        case 5:
            return Qt.vector3d(0.04, 0.18, 0.03)
        case 3:
            return Qt.vector3d(0.12, 0.03, 0.18)
        case 4:
            return Qt.vector3d(0.00, 0.18, 0.22)
        case 6:
            return Qt.vector3d(0.22, 0.12, 0.02)
        default:
            return Qt.vector3d(0.02, 0.02, 0.02)
        }
    }

    function dominoPipLayout(count, halfLengthMm, faceWidthMm) {
        const c = Number(count)
        const y = halfLengthMm * 0.25
        const w = faceWidthMm * 0.25
        if (c === 0)
            return []
        if (c === 1)
            return [{ "y": 0, "w": 0 }]
        if (c === 2)
            return [{ "y": -y, "w": -w }, { "y": y, "w": w }]
        if (c === 3)
            return [{ "y": -y, "w": -w }, { "y": 0, "w": 0 }, { "y": y, "w": w }]
        if (c === 4)
            return [{ "y": -y, "w": -w }, { "y": y, "w": -w }, { "y": -y, "w": w }, { "y": y, "w": w }]
        if (c === 5)
            return [{ "y": -y, "w": -w }, { "y": y, "w": -w }, { "y": 0, "w": 0 }, { "y": -y, "w": w }, { "y": y, "w": w }]
        if (c === 6)
            return [
                { "y": -y, "w": -w }, { "y": y, "w": -w },
                { "y": -y, "w": 0 },  { "y": y, "w": 0 },
                { "y": -y, "w": w },  { "y": y, "w": w }
            ]
        return []
    }

    function wrapDegrees(value) {
        let wrapped = value % 360
        if (wrapped < -180)
            wrapped += 360
        if (wrapped > 180)
            wrapped -= 360
        return wrapped
    }

    function moveScene(forwardMm, strafeMm) {
        const headingRad = degreesToRadians(sceneYawDeg)
        const forwardX = Math.cos(headingRad)
        const forwardY = Math.sin(headingRad)
        const strafeX = -Math.sin(headingRad)
        const strafeY = Math.cos(headingRad)
        const offsetX = forwardMm * forwardX + strafeMm * strafeX
        const offsetY = forwardMm * forwardY + strafeMm * strafeY
        scenePosition = Qt.vector3d(scenePosition.x + offsetX,
                                     scenePosition.y + offsetY,
                                     scenePosition.z)
        updateCamera()
    }

    function yaw(deltaDeg) {
        sceneYawDeg = wrapDegrees(sceneYawDeg + deltaDeg)
        updateCamera()
    }

    function pitch(deltaDeg) {
        cameraPitchDeg = clamp(cameraPitchDeg + deltaDeg, minPitchDeg, maxPitchDeg)
        updateCamera()
    }

    function zoom(deltaGl) {
        cameraDistanceGl = clamp(cameraDistanceGl + deltaGl, minZoomGl, maxZoomGl)
        updateCamera()
    }

    function resetCamera() {
        scenePosition = Qt.vector3d(0, 0, 0)
        sceneYawDeg = 90
        cameraPitchDeg = -30
        cameraDistanceGl = 10
        updateCamera()
    }

    function elevate(deltaMm) {
        scenePosition = Qt.vector3d(scenePosition.x,
                                     scenePosition.y,
                                     scenePosition.z + deltaMm)
        updateCamera()
    }

    function zoomIn() { zoom(-zoomStepGl) }
    function zoomOut() { zoom(zoomStepGl) }
    function pitchUp() { pitch(-pitchStepDeg) }
    function pitchDown() { pitch(pitchStepDeg) }
    function toggleAxes() { showAxes = !showAxes }

    function updateCamera() {
        if (!worldCamera)
            return

        const clampedDistance = clamp(cameraDistanceGl, minZoomGl, maxZoomGl)
        if (clampedDistance !== cameraDistanceGl)
            cameraDistanceGl = clampedDistance

        const yawRad = degreesToRadians(sceneYawDeg)
        const pitchRad = degreesToRadians(cameraPitchDeg)
        const distanceMm = clampedDistance / worldScale
        const cosPitch = Math.cos(pitchRad)
        const sinPitch = Math.sin(pitchRad)
        const cosYaw = Math.cos(yawRad)
        const sinYaw = Math.sin(yawRad)

        const relX = -cosPitch * cosYaw * distanceMm
        const relY = -cosPitch * sinYaw * distanceMm
        const relZ = -sinPitch * distanceMm

        const focusGl = worldToSceneVector(scenePosition.x, scenePosition.y, scenePosition.z)
        const offsetGl = worldToSceneVector(relX, relY, relZ)

        cameraPosition = Qt.vector3d(
            focusGl.x + offsetGl.x,
            focusGl.y + offsetGl.y,
            focusGl.z + offsetGl.z
        )

        if (worldCamera.lookAt) {
            worldCamera.lookAt(focusGl)
        } else {
            worldCamera.eulerRotation = Qt.vector3d(cameraPitchDeg, sceneYawDeg, 0)
        }
    }

    focus: true
    Component.onCompleted: {
        worldView.forceActiveFocus()
        updateCamera()
    }

    environment: SceneEnvironment {
        clearColor: Qt.rgba(0.1, 0.1, 0.12, 1.0)
        backgroundMode: SceneEnvironment.Color
        antialiasingMode: SceneEnvironment.MSAA
        antialiasingQuality: SceneEnvironment.High
    }

    camera: PerspectiveCamera {
        id: worldCamera
        objectName: "worldCamera"
        fieldOfView: 50
        clipNear: 0.1
        clipFar: 200
        position: cameraPosition
    }

    Node {
        id: sceneRoot
        objectName: "sceneRoot"
        position: Qt.vector3d(0, 0, 0)

        Node {
            id: sceneBasis
            rotation: Qt.quaternion(0.7071, -0.7071, 0, 0)
            scale: Qt.vector3d(worldScale, worldScale, worldScale)

            Node {
                id: sceneFrame
                objectName: "sceneFrame"

                DirectionalLight {
                    brightness: 0.3
                    ambientColor: Qt.rgba(0.4, 0.4, 0.4, 1)
                    eulerRotation: Qt.vector3d(0, 0, 0)
                    castsShadow: false
                }

                DirectionalLight {
                    brightness: 0.6
                    eulerRotation: Qt.vector3d(-50, 35, 0)
                    castsShadow: false
                }

                DirectionalLight {
                    brightness: 0.4
                    eulerRotation: Qt.vector3d(20, -145, 0)
                    castsShadow: false
                }

                Model {
                    id: ground
                    source: "#Cube"
                    position: Qt.vector3d(0, 0, -25)
                    scale: Qt.vector3d(200, 200, 0.5)
                    materials: PrincipledMaterial {
                        baseColor: Qt.rgba(0.25, 0.25, 0.25, 1)
                        roughness: 0.9
                        metalness: 0.0
                    }
                }

                Node {
                    id: floorGrid
                    visible: true
                    
                    Repeater3D {
                        model: 61
                        Model {
                            source: "#Cube"
                            property real yPosMm: (index - 30) * 100
                            position: Qt.vector3d(0, yPosMm, 0)
                            scale: Qt.vector3d(60, 0.02, 0.01)
                            materials: PrincipledMaterial {
                                baseColor: Qt.rgba(0.65, 0.65, 0.65, 1)
                                roughness: 0.5
                                metalness: 0.0
                                cullMode: Material.NoCulling
                            }
                        }
                    }
                    
                    Repeater3D {
                        model: 61
                        Model {
                            source: "#Cube"
                            property real xPosMm: (index - 30) * 100
                            position: Qt.vector3d(xPosMm, 0, 0)
                            scale: Qt.vector3d(0.02, 60, 0.01)
                            materials: PrincipledMaterial {
                                baseColor: Qt.rgba(0.65, 0.65, 0.65, 1)
                                roughness: 0.5
                                metalness: 0.0
                                cullMode: Material.NoCulling
                            }
                        }
                    }
                }

                Node {
                    id: axes
                    visible: showAxes
                    readonly property real axisLengthMm: 100
                    readonly property real axisThicknessMm: 2

                    Model {
                        source: "#Cube"
                        scale: Qt.vector3d(axes.axisLengthMm / 100, axes.axisThicknessMm / 100, axes.axisThicknessMm / 100)
                        position: Qt.vector3d(50, 0, 0)
                        materials: PrincipledMaterial {
                            baseColor: "#ff0000"
                            cullMode: Material.NoCulling
                        }
                    }

                    Model {
                        source: "#Cube"
                        scale: Qt.vector3d(axes.axisThicknessMm / 100, axes.axisLengthMm / 100, axes.axisThicknessMm / 100)
                        position: Qt.vector3d(0, 50, 0)
                        materials: PrincipledMaterial {
                            baseColor: "#00ff00"
                            cullMode: Material.NoCulling
                        }
                    }

                    Model {
                        source: "#Cube"
                        scale: Qt.vector3d(axes.axisThicknessMm / 100, axes.axisThicknessMm / 100, axes.axisLengthMm / 100)
                        position: Qt.vector3d(0, 0, 50)
                        materials: PrincipledMaterial {
                            baseColor: "#4770f2"
                            cullMode: Material.NoCulling
                        }
                    }
                }
            }
        }
    }

    Component {
        id: robotDelegate
        Node {
            property var model
            parent: sceneFrame
            visible: !model.missing
            readonly property real radiusMm: 32
            readonly property real heightMm: 72
            position: Qt.vector3d(model.x, model.y, model.z + heightMm / 2)
            eulerRotation.z: radiansToDegrees(model.theta)

            Model {
                source: "#Cylinder"
                eulerRotation.x: 90
                scale: Qt.vector3d(radiusMm / 50, heightMm / 100, radiusMm / 50)
                materials: PrincipledMaterial {
                    baseColor: "#a6a6a6"
                    roughness: 0.4
                    metalness: 0.1
                    cullMode: Material.NoCulling
                }
            }

            Model {
                source: "#Sphere"
                position: Qt.vector3d(30, 0, 42 - heightMm / 2)
                scale: Qt.vector3d(0.24, 0.24, 0.24)
                materials: PrincipledMaterial {
                    baseColor: "#000000"
                    roughness: 0.3
                    metalness: 0.0
                    cullMode: Material.NoCulling
                }
            }
        }
    }

    Component {
        id: ballDelegate
        Node {
            property var model
            parent: sceneFrame
            visible: !model.missing
            readonly property real radiusMm: (model.diameter_mm || 25) / 2
            position: Qt.vector3d(model.x, model.y, model.z)
            eulerRotation.z: radiansToDegrees(model.theta)

            Model {
                source: "#Sphere"
                scale: Qt.vector3d(parent.radiusMm * 2 / 100, parent.radiusMm * 2 / 100, parent.radiusMm * 2 / 100)
                materials: PrincipledMaterial {
                    baseColor: model.visible ? "#e6b319" : "#8a6b0f"
                    roughness: 0.3
                    cullMode: Material.NoCulling
                    emissiveFactor: model.visible ? Qt.vector3d(0.1, 0.08, 0.01) : Qt.vector3d(0, 0, 0)
                }
            }
        }
    }

    Component {
        id: barrelDelegate
        Node {
            property var model
            parent: sceneFrame
            visible: !model.missing
            readonly property real radiusMm: (model.diameter_mm || 22) / 2
            readonly property real heightMm: model.height_mm || 25
            position: Qt.vector3d(model.x, model.y, model.z)
            eulerRotation.z: radiansToDegrees(model.theta)

            Model {
                source: "#Cylinder"
                eulerRotation.x: 90
                scale: Qt.vector3d(parent.radiusMm / 50, parent.heightMm / 100, parent.radiusMm / 50)
                materials: PrincipledMaterial {
                    baseColor: {
                        if (model.type === "barrel_orange")
                            return model.visible ? "#ff8010" : "#994d0a"
                        if (model.type === "barrel_blue")
                            return model.visible ? "#4770f2" : "#2a4391"
                        return model.visible ? "#18d0ff" : "#0e7e97"
                    }
                    roughness: 0.45
                    cullMode: Material.NoCulling
                    emissiveFactor: {
                        if (!model.visible)
                            return Qt.vector3d(0, 0, 0)
                        if (model.type === "barrel_orange")
                            return Qt.vector3d(0.15, 0.08, 0.01)
                        if (model.type === "barrel_blue")
                            return Qt.vector3d(0.05, 0.08, 0.18)
                        return Qt.vector3d(0.06, 0.2, 0.25)
                    }
                }
            }
        }
    }

    Component {
        id: dominoDelegate
        Node {
            id: dominoRoot
            property var model
            parent: sceneFrame
            visible: !model.missing

            readonly property bool isFallen: Boolean(model.is_fallen)

            readonly property real depthMm: isFallen ? 24 : (model.thickness_mm || 8)
            readonly property real lengthMm: model.width_mm || 48
            readonly property real heightMm: isFallen ? 8 : (model.height_mm || 24)

            readonly property real halfLengthMm: lengthMm / 2
            readonly property real pipRadiusMm: Math.min(isFallen ? depthMm : heightMm, halfLengthMm) * 0.12
            
            readonly property real activeSurfaceOffset: isFallen ? (heightMm / 2 + 0.1) : (depthMm / 2 + 0.1)

            readonly property var dominoHalves: (model.domino_halves && model.domino_halves.length)
                                                ? model.domino_halves
                                                : [{ "count": null, "local_y": -lengthMm / 4 },
                                                   { "count": null, "local_y": lengthMm / 4 }]

            position: Qt.vector3d(model.x, model.y, model.z)
            eulerRotation.z: radiansToDegrees(model.theta) - 90

            // Main Cuboid Block
            Model {
                source: "#Cube"
                scale: Qt.vector3d(depthMm / 100, lengthMm / 100, heightMm / 100)
                materials: PrincipledMaterial {
                    baseColor: model.visible ? "#f4f0dc" : "#8e8a76"
                    emissiveFactor: model.visible ? Qt.vector3d(0.03, 0.025, 0.015) : Qt.vector3d(0, 0, 0)
                    roughness: 0.7
                    metalness: 0.0
                    cullMode: Material.NoCulling
                }
            }

            // Center Divider Line
            Model {
                source: "#Cube"
                position: dominoRoot.isFallen
                          ? Qt.vector3d(0, 0, dominoRoot.activeSurfaceOffset)
                          : Qt.vector3d(dominoRoot.activeSurfaceOffset, 0, 0)
                scale: dominoRoot.isFallen
                       ? Qt.vector3d((dominoRoot.depthMm * 0.85) / 100, 1.0 / 100, 0.2 / 100)
                       : Qt.vector3d(0.2 / 100, 1.0 / 100, (dominoRoot.heightMm * 0.85) / 100)
                materials: PrincipledMaterial {
                    baseColor: model.visible ? "#1a1a1a" : "#444444"
                    roughness: 0.4
                    metalness: 0.0
                    cullMode: Material.NoCulling
                }
            }

            // Pips Rendered Across Active Face
            Repeater3D {
                model: dominoHalves

                Node {
                    id: halfNode
                    property var halfData: modelData || ({ "count": null, "local_y": 0 })
                    property int halfCount: halfData.count === null || halfData.count === undefined ? -1 : Number(halfData.count)
                    property real halfCenterY: Number(halfData.local_y) || 0
                    property var pipSpecs: dominoPipLayout(halfCount, dominoRoot.halfLengthMm, dominoRoot.isFallen ? dominoRoot.depthMm : dominoRoot.heightMm)

                    position: Qt.vector3d(0, halfCenterY, 0)

                    Image {
                        id: unknownDominoImage
                        source: "image://dominotext/question"
                        visible: false
                        cache: true
                    }

                    Model {
                        visible: halfCount < 0
                        source: "#Cube"
                        position: dominoRoot.isFallen
                                  ? Qt.vector3d(0, 0, dominoRoot.activeSurfaceOffset)
                                  : Qt.vector3d(dominoRoot.activeSurfaceOffset, 0, 0)
                        scale: dominoRoot.isFallen
                               ? Qt.vector3d((dominoRoot.depthMm * 0.68) / 100, (dominoRoot.halfLengthMm * 0.56) / 100, 0.1 / 100)
                               : Qt.vector3d(0.1 / 100, (dominoRoot.halfLengthMm * 0.56) / 100, (dominoRoot.heightMm * 0.68) / 100)
                        materials: PrincipledMaterial {
                            baseColor: Qt.rgba(1, 1, 1, 1)
                            baseColorMap: Texture {
                                sourceItem: unknownDominoImage
                            }
                            alphaMode: PrincipledMaterial.Blend
                            emissiveFactor: Qt.vector3d(0.02, 0.02, 0.02)
                            roughness: 0.35
                            cullMode: Material.NoCulling
                        }
                    }

                    Repeater3D {
                        model: pipSpecs

                        Node {
                            id: pipNode
                            property var pipSpec: modelData
                            property real pipOffsetY: Number(pipSpec.y) || 0
                            property real pipOffsetSecondary: Number(pipSpec.w) || 0

                            Model {
                                source: "#Cylinder"
                                eulerRotation: dominoRoot.isFallen ? Qt.vector3d(-90, 0, 0) : Qt.vector3d(0, 0, 90)
                                position: dominoRoot.isFallen
                                          ? Qt.vector3d(pipOffsetSecondary, pipOffsetY, dominoRoot.activeSurfaceOffset)
                                          : Qt.vector3d(dominoRoot.activeSurfaceOffset, pipOffsetY, pipOffsetSecondary)
                                scale: dominoRoot.isFallen
                                       ? Qt.vector3d(dominoRoot.pipRadiusMm / 50,
                                                      0.1 / 100,
                                                      dominoRoot.pipRadiusMm / 50)
                                       : Qt.vector3d(dominoRoot.pipRadiusMm / 50,
                                                      0.1 / 100,
                                                      dominoRoot.pipRadiusMm / 50)
                                materials: PrincipledMaterial {
                                    baseColor: dominoPipColor(halfNode.halfCount)
                                    emissiveFactor: dominoPipEmissive(halfNode.halfCount)
                                    roughness: 0.2
                                    cullMode: Material.NoCulling
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    Component {
        id: markerDelegate
        Node {
            property var model
            parent: sceneFrame
            visible: !model.missing
            readonly property real widthMm: model.width_mm || 38
            readonly property real heightMm: model.height_mm || 48
            readonly property real thicknessMm: model.thickness_mm || 2
            position: Qt.vector3d(model.x, model.y, model.z)
            eulerRotation.z: radiansToDegrees(model.theta)

            Model {
                source: "#Cube"
                scale: Qt.vector3d(thicknessMm / 100, widthMm / 100, heightMm / 100)
                materials: PrincipledMaterial {
                    baseColor: {
                        if (model.type === "apriltag")
                            return "#804ce6";
                        if (model.type === "aruco")
                            return model.visible ? "#00ff00" : "#1f431f";
                        return "#202020";
                    }
                    emissiveFactor: {
                        if (!model.visible)
                            return Qt.vector3d(0.02, 0.02, 0.02);
                        if (model.type === "apriltag")
                            return Qt.vector3d(0.08, 0.05, 0.15);
                        if (model.type === "aruco")
                            return Qt.vector3d(0.08, 0.16, 0.08);
                        return Qt.vector3d(0.1, 0.1, 0.1);
                    }
                    roughness: 0.2
                    cullMode: Material.NoCulling
                }
            }

            Image {
                id: tagImageFront
                source: (model.type === "apriltag" || model.type === "aruco")
                        ? "image://tagtexture/" + (model.type === "aruco" ? "aruco-" : "") + String(model.marker_id)
                        : ""
                visible: false
                cache: true
            }

            Image {
                id: tagImageBack
                source: (model.type === "apriltag" || model.type === "aruco")
                        ? "image://tagtexture/back-" + (model.type === "aruco" ? "aruco-" : "") + String(model.marker_id)
                        : ""
                visible: false
                cache: true
            }
            
            Model {
                id: textPanelFront
                visible: (model.type === "apriltag" || model.type === "aruco")
                         && model.marker_id !== null && model.marker_id !== undefined && model.marker_id !== ""
                source: "#Cube"
                scale: Qt.vector3d(0.005, heightMm / 100, widthMm / 100)
                position: Qt.vector3d(thicknessMm / 2 + 0.5, 0, 0)
                eulerRotation: Qt.vector3d(90, 0, 0)
                
                materials: PrincipledMaterial {
                    baseColor: "#ffffff"
                    baseColorMap: Texture {
                        sourceItem: tagImageFront
                    }
                    emissiveFactor: model.visible ? Qt.vector3d(0.1, 0.06, 0.2) : Qt.vector3d(0, 0, 0)
                    roughness: 0.1
                    cullMode: Material.NoCulling
                }
                opacity: model.visible ? 1.0 : 0.8
            }
            
            Model {
                id: textPanelBack
                visible: (model.type === "apriltag" || model.type === "aruco")
                         && model.marker_id !== null && model.marker_id !== undefined && model.marker_id !== ""
                source: "#Cube"
                scale: Qt.vector3d(0.005, heightMm / 100, widthMm / 100)
                position: Qt.vector3d(-thicknessMm / 2 - 0.5, 0, 0)
                eulerRotation: Qt.vector3d(90, 0, 0)
                
                materials: PrincipledMaterial {
                    baseColor: "#ffffff"
                    baseColorMap: Texture {
                        sourceItem: tagImageBack
                    }
                    emissiveFactor: model.visible ? Qt.vector3d(0.1, 0.06, 0.2) : Qt.vector3d(0, 0, 0)
                    roughness: 0.1
                    cullMode: Material.NoCulling
                }
                opacity: model.visible ? 1.0 : 0.8
            }
        }
    }

    Component {
        id: wallDelegate
        Node {
            property var model
            parent: sceneFrame
            visible: !model.missing
            readonly property real lengthMm: model.length_mm || 300
            readonly property real heightMm: model.height_mm || 210
            readonly property real thicknessMm: model.thickness_mm || 4
            readonly property var doorways: (model.doorways && model.doorways.length) ? model.doorways : []
            readonly property real wallBaseLocalZ: -heightMm / 2
            readonly property real doorHeightMm: {
                if (!doorways.length)
                    return 0
                var maxHeight = 0
                for (var i = 0; i < doorways.length; ++i) {
                    var h = Number(doorways[i].height) || 0
                    if (h > maxHeight)
                        maxHeight = h
                }
                return Math.max(0, Math.min(heightMm, maxHeight))
            }
            readonly property real lowerHeightMm: doorHeightMm > 0 ? doorHeightMm : heightMm
            readonly property var lowerSegments: computeLowerSegments()
            position: Qt.vector3d(model.x, model.y, model.z)
            eulerRotation.z: radiansToDegrees(model.theta)

            function computeLowerSegments() {
                if (!doorways.length) {
                    return [{ "center": 0, "width": lengthMm }]
                }

                var cursor = -lengthMm / 2
                var end = lengthMm / 2
                var segments = []
                var sorted = []
                for (var i = 0; i < doorways.length; ++i)
                    sorted.push(doorways[i])
                sorted.sort(function(a, b) { return Number(a.x) - Number(b.x) })

                for (var j = 0; j < sorted.length; ++j) {
                    var spec = sorted[j]
                    var centerY = (Number(spec.x) || 0) - lengthMm / 2
                    var width = Math.max(0, Number(spec.width) || 0)
                    var left = centerY - width / 2
                    var right = centerY + width / 2
                    if (left > cursor) {
                        var segWidth = left - cursor
                        segments.push({
                            "center": cursor + segWidth / 2,
                            "width": segWidth
                        })
                    }
                    cursor = Math.max(cursor, right)
                }

                if (cursor < end) {
                    var tailWidth = end - cursor
                    segments.push({
                        "center": cursor + tailWidth / 2,
                        "width": tailWidth
                    })
                }
                return segments
            }

            PrincipledMaterial {
                id: wallMaterial
                baseColor: model.visible ? Qt.rgba(0.88, 0.78, 0.22, 0.74) : Qt.rgba(0.55, 0.48, 0.14, 0.60)
                alphaMode: PrincipledMaterial.Blend
                roughness: 0.65
                cullMode: Material.NoCulling
                emissiveFactor: model.visible ? Qt.vector3d(0.06, 0.06, 0.02) : Qt.vector3d(0.02, 0.02, 0.01)
            }

            Repeater3D {
                model: lowerSegments.length
                Model {
                    property int idx: index
                    source: "#Cube"
                    property var seg: lowerSegments[idx]
                    position: Qt.vector3d(0, Number(seg.center) || 0, wallBaseLocalZ + lowerHeightMm / 2)
                    scale: Qt.vector3d(thicknessMm / 100, (Number(seg.width) || 0) / 100, lowerHeightMm / 100)
                    materials: wallMaterial
                }
            }

            Model {
                visible: doorHeightMm > 0 && doorHeightMm < heightMm
                source: "#Cube"
                property real transomHeightMm: Math.max(0, heightMm - doorHeightMm)
                position: Qt.vector3d(0, 0, wallBaseLocalZ + doorHeightMm + transomHeightMm / 2)
                scale: Qt.vector3d(thicknessMm / 100, lengthMm / 100, transomHeightMm / 100)
                materials: wallMaterial
            }
        }
    }

    Component {
        id: worldDelegate
        Node {
            id: worldDelegateRoot
            property var modelSnapshot: model
            property Node delegateItem
            property Component delegateComponent: null

            function componentForType(typeName) {
                switch (typeName) {
                case "robot":
                    return robotDelegate
                case "sports_ball":
                    return ballDelegate
                case "barrel":
                case "barrel_orange":
                case "barrel_blue":
                    return barrelDelegate
                case "domino":
                    return dominoDelegate
                case "apriltag":
                case "aruco":
                    return markerDelegate
                case "wall":
                    return wallDelegate
                default:
                    return null
                }
            }

            function rebuild() {
                const data = modelSnapshot
                if (!data || !data.type) {
                    if (delegateItem) {
                        delegateItem.destroy()
                        delegateItem = null
                    }
                    delegateComponent = null
                    return
                }

                const component = componentForType(data.type)
                if (delegateComponent === component && delegateItem) {
                    if (delegateItem.model !== data)
                        delegateItem.model = data
                    return
                }

                if (delegateItem) {
                    delegateItem.destroy()
                    delegateItem = null
                }
                delegateComponent = component
                if (!component)
                    return

                delegateItem = component.createObject(sceneFrame, {
                    "model": data
                })
                if (!delegateItem)
                    console.warn("Failed to create delegate for type", data.type)
            }

            onModelSnapshotChanged: rebuild()
            Component.onDestruction: {
                if (delegateItem) {
                    delegateItem.destroy()
                    delegateItem = null
                }
                delegateComponent = null
            }
        }
    }

    Repeater3D {
        id: worldRepeater
        parent: sceneFrame
        model: worldModel
        delegate: worldDelegate
    }

    Keys.onPressed: function(event) {
        switch (event.key) {
        case Qt.Key_W:
            moveScene(moveStepMm, 0)
            event.accepted = true
            break
        case Qt.Key_S:
            moveScene(-moveStepMm, 0)
            event.accepted = true
            break
        case Qt.Key_A:
            moveScene(0, moveStepMm)
            event.accepted = true
            break
        case Qt.Key_D:
            moveScene(0, -moveStepMm)
            event.accepted = true
            break
        case Qt.Key_J:
            yaw(-yawStepDeg)
            event.accepted = true
            break
        case Qt.Key_L:
            yaw(yawStepDeg)
            event.accepted = true
            break
        case Qt.Key_Left:
            yaw(-yawStepDeg)
            event.accepted = true
            break
        case Qt.Key_Right:
            yaw(yawStepDeg)
            event.accepted = true
            break
        case Qt.Key_I:
            pitchUp()
            event.accepted = true
            break
        case Qt.Key_K:
            pitchDown()
            event.accepted = true
            break
        case Qt.Key_Up:
            pitchUp()
            event.accepted = true
            break
        case Qt.Key_Down:
            pitchDown()
            event.accepted = true
            break
        case Qt.Key_Comma:
        case Qt.Key_Less:
            zoomIn()
            event.accepted = true
            break
        case Qt.Key_Period:
        case Qt.Key_Greater:
            zoomOut()
            event.accepted = true
            break
        case Qt.Key_Minus:
        case Qt.Key_Underscore:
            zoomOut()
            event.accepted = true
            break
        case Qt.Key_Equal:
        case Qt.Key_Plus:
            zoomIn()
            event.accepted = true
            break
        case Qt.Key_PageUp:
            elevate(elevateStepMm)
            event.accepted = true
            break
        case Qt.Key_PageDown:
            elevate(-elevateStepMm)
            event.accepted = true
            break
        case Qt.Key_Q:
            elevate(elevateStepMm)
            event.accepted = true
            break
        case Qt.Key_E:
            elevate(-elevateStepMm)
            event.accepted = true
            break
        case Qt.Key_Z:
            resetCamera()
            event.accepted = true
            break
        case Qt.Key_X:
            toggleAxes()
            event.accepted = true
            break
        case Qt.Key_H:
            if (typeof viewerApp !== "undefined" && viewerApp && viewerApp.printHelp) {
                viewerApp.printHelp()
            }
            event.accepted = true
            break
        default:
            break
        }
    }

    MouseArea {
        anchors.fill: parent
        acceptedButtons: Qt.LeftButton
        onPressed: worldView.forceActiveFocus()
    }
}