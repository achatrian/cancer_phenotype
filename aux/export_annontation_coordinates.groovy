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
def pathOutput = QPEx.buildFilePath(QPEx.PROJECT_BASE_DIR, 'polygons')
QPEx.mkdirs(pathOutput)

String name = server.getShortServerName()
def file = new File(pathOutput, name + '.txt')
file.text = ''

// Export each annotation
for (pathObject in getAllObjects()) {
    // Check for interrupt (Run -> Kill running script)
    if (Thread.interrupted())
        break
    // Get the ROI
    def roi = pathObject.getROI()
    if (roi == null)
        continue
    // Get the class name:
    def pathClass = pathObject.getPathClass()
    // Write the points; but beware areas, and also ellipses!
    file << pathClass << ';' << roi.getPolygonPoints() << System.lineSeparator()
}
print 'Done!'
