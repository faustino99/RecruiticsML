package com.recruitics.ml

import org.junit.jupiter.api.Test


internal class SparkMLTests {

  @Test
  internal fun `test functions`() {
    cleanAndExport()
    /*generalizedLinearRegressionTest()     // the better one (linear regression model)
    desicionTreeTest()*/                    // decision tree model w/ less accuracy
  }
}
