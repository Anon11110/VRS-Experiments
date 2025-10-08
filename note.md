We wanted to test out different ways to fake 4x4 Variable Rate Shading (VRS) in a post-processing step. The main goal was to see how our software tricks stacked up against what the actual hardware does.
Our method was pretty simple: we built a simulator that runs after the main render. It looks at the final image, grabs one pixel's color from a 4x4 block, and just copies it to the other 15 pixels in that block.
It's important to remember this only really works for post-processing. For regular rendering, real hardware VRS does a bunch of complex stuff like recalculating shader inputs (barycentrics and things) at a new sample spot. Our simulator can't do that; it just copies colors that are already there. But for post-fx passes that use texelFetch, our simulation is actually a perfect match for how the hardware behaves.
How We Set Everything Up
To make sure our tests were fair, we needed a good testing ground.
Test Case: The "Wildlife" DoF Pass: We picked this pass because it uses texelFetch, which meant our simulator could perfectly match the hardware. We ran a test and confirmed that our simulated 2x2 VRS looked identical to the hardware 2x2 VRS. This gave us confidence that our setup was solid for testing the 4x4 policies.
The "AZH" Blooming Pass: We wanted to use this one too, but RenderDoc kept giving us a black screen when we tried to grab the framebuffer. We'll have to circle back to this one later.
We couldn't get real 4x4 VRS to turn on (seems like the hardware/drivers don't support it), so we made our own baseline to compare against: a simulated 4x4 VRS that just picks the color from the centermost pixel. This is probably the simplest thing the hardware would do.
The Different 4x4 Policies We Tried
We cooked up a few different strategies for picking that one important color out of the 16 pixels:
Nearest-Neighbor on Centroid (Our Baseline): Just grabs the color from the middle pixel.
Center Sample with Bilinear: Samples right in the geometric center of the block, letting the hardware's bilinear filtering blend the 4 closest pixels.
Corner Cycling: Picks a corner to sample from, cycling through the four corners in a pattern.
Content-Adaptive Corner: Checks the gradient (how much the color is changing) at all four corners and picks the corner where the color is the most stable.
Dynamic Gradient Centroid: A fancier method that finds a "center of visual activity" based on gradients and samples there.
Minimum/Maximum Gradient: Finds the pixel with the least or most amount of color change across the entire 16-pixel block and samples it.
What We Found Out
Here's the main takeaway: Only the "Center Sample with Bilinear" policy looked better than the hardware's 2x2 VRS.
Even though it was an improvement, it still didn't look as good as just rendering at native resolution.
This shows that letting the hardware do bilinear filtering from a floating-point coordinate is a big deal. Averaging those four central pixels gives you a much nicer, more stable color than just picking one pixel, which can easily be an outlier and create ugly artifacts.
We also realized that the Wildlife pass was easy to simulate because it uses texelFetch. Passes that use regular bilinear samplers (like the AZH bloom) are kind of already doing their own "center sampling," so it'll be interesting to see how our tests work there.
What's Next?
Simple: we need to get the AZH blooming pass working. Testing on a pass that uses standard bilinear filtering is the next logical step. It'll show us how our policies behave when they're fighting against the filtering the hardware is already doing.
