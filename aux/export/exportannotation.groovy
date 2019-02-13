/**
 * Script to export binary masks corresponding to all annotations of an image,
 * optionally along with extracted image regions.
 *
 * Note: Pay attention to the 'downsample' value to control the export resolution!
 *
 * @author Pete Bankhead
 */

import qupath.lib.images.servers.ImageServer
import qupath.lib.objects.PathObject
import qupath.lib.regions.RegionRequest
import qupath.lib.roi.PathROIToolsAwt
import qupath.lib.scripting.QPEx

import javax.imageio.ImageIO
import java.awt.Color
import java.awt.image.BufferedImage

// Get the main QuPath data structures
def imageData = QPEx.getCurrentImageData()
def hierarchy = imageData.getHierarchy()
def server = imageData.getServer()

// Request all objects from the hierarchy & filter only the annotations
def annotations = hierarchy.getFlattenedObjectList(null).findAll {it.isAnnotation()}

// Define downsample value for export resolution & output directory, creating directory if necessary
def downsample = 1.0 // !!!! halving resolution
def pathOutput = QPEx.buildFilePath('/Volumes/A-CH-EXDISK/Projects/Project_annotate', 'masks')
//def pathOutput = QPEx.buildFilePath(QPEx.PROJECT_BASE_DIR, 'masks')
QPEx.mkdirs(pathOutput)

// Define image export type; valid values are JPG, PNG or null (if no image region should be exported with the mask)
// Note: masks will always be exported as PNG
def imageExportType = 'JPG'

// Export each annotation
annotations.each {
    saveAnnotations(pathOutput, server, it, downsample, imageExportType)
}

saveDetections(pathOutput, server, "Nucleus", downsample, imageExportType)

print 'Done!'

//-------------------------------------------------------------------------------------

// Saves Annotations. Some of which are rectangles and some of which are nuclei
def saveAnnotations(String pathOutput, ImageServer server, PathObject pathObject, double downsample, String imageExportType) {
    def roi = pathObject.getROI()
    def pathClass = pathObject.getPathClass()
    def classificationName = pathClass == null ? 'None' : pathClass.toString()
    if (roi == null) {
        print 'Warning! No ROI for object ' + pathObject + ' - cannot export corresponding region & mask'
        return
    }

    // Create a region from the ROI
    //print roi.getPolygonPoints()
    /*print(i)
    print roi.getPolygonPoints().x.toArray()
    print roi.getPolygonPoints().y.toArray()
    print Math.floor(server.getHeight())*/

    def region = RegionRequest.createInstance(server.getPath(), downsample, roi)

    // Create a name
    String name = String.format('%s_%s_(%.2f,%d,%d,%d,%d)',
            server.getShortServerName(),
            classificationName,
            region.getDownsample(),
            region.getX(),
            region.getY(),
            region.getWidth(),
            region.getHeight()
    )

    // Request the BufferedImage
    def img = server.readBufferedImage(region)

    // Create a mask using Java2D functionality
    // (This involves applying a transform to a graphics object, so that none needs to be applied to the ROI coordinates)
    def shape = PathROIToolsAwt.getShape(roi)
    def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
    def g2d = imgMask.createGraphics()
    g2d.setColor(Color.WHITE)
    g2d.scale(1.0/downsample, 1.0/downsample)
    g2d.translate(-region.getX(), -region.getY())
    g2d.fill(shape)
    g2d.dispose()

    // Create filename & export
    if (imageExportType != null) {
        def fileImage = new File(pathOutput, name + '.' + imageExportType.toLowerCase())
        ImageIO.write(img, imageExportType, fileImage)
    }
    // Export the mask
    def fileMask = new File(pathOutput, name + '-mask.png')
    ImageIO.write(imgMask, 'PNG', fileMask)

}

//-------------------------------------------------------------------------------------

// Save Detection. All detections are nuclei
def saveDetections(String pathOutput, ImageServer server,String detection_type, double downsample, String imageExportType) {

    for (detection in getDetectionObjects()) {

        roi = detection.getROI()

        if (roi == null) {
            print 'Warning! No ROI for object ' + detection_type + ' - cannot export corresponding region & mask'
            return
        }

        def region = RegionRequest.createInstance(server.getPath(), downsample, roi)

        // Create a name
        String name = String.format('%s_%s_(%.2f,%d,%d,%d,%d)',
                server.getShortServerName(),
                detection_type,
                region.getDownsample(),
                region.getX(),
                region.getY(),
                region.getWidth(),
                region.getHeight()
        )

        // Request the BufferedImage
        def img = server.readBufferedImage(region)

        // Create a mask using Java2D functionality
        // (This involves applying a transform to a graphics object, so that none needs to be applied to the ROI coordinates)
        def shape = PathROIToolsAwt.getShape(roi)
        def imgMask = new BufferedImage(img.getWidth(), img.getHeight(), BufferedImage.TYPE_BYTE_GRAY)
        def g2d = imgMask.createGraphics()
        g2d.setColor(Color.WHITE)
        g2d.scale(1.0/downsample, 1.0/downsample)
        g2d.translate(-region.getX(), -region.getY())
        g2d.fill(shape)
        g2d.dispose()

        // Create filename & export
        if (imageExportType != null) {
            def fileImage = new File(pathOutput, name + '.' + imageExportType.toLowerCase())
            ImageIO.write(img, imageExportType, fileImage)
        }
        // Export the mask
        def fileMask = new File(pathOutput, name + '-mask.png')
        ImageIO.write(imgMask, 'PNG', fileMask)
    }
}
