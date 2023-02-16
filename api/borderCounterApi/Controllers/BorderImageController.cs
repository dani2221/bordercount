using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using borderCounterApi.DTOs.BorderImage;
using borderCounterApi.Services;
using Microsoft.AspNetCore.Mvc;

// For more information on enabling Web API for empty projects, visit https://go.microsoft.com/fwlink/?LinkID=397860

namespace borderCounterApi.Controllers
{
    [Route("api/[controller]")]
    public class BorderImageController : Controller
    {

        private readonly IBorderImageService borderImageService;

        public BorderImageController(IBorderImageService borderImageService)
        {
            this.borderImageService = borderImageService;
        }

        // GET api/BorderImage/medzitlija
        [HttpGet("{border}")]
        public string Get(string border)
        {
            return borderImageService.GetImage(border);
        }

        // POST api/BorderImage
        [HttpPost]
        public void Post([FromBody]BorderImageRequest request)
        {
            borderImageService.SaveImage(request);
        }
    }
}

